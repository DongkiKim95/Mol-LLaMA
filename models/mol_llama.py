"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from peft import get_peft_model, LoraConfig, TaskType

from utils.configuration_mol_llama import MolLLaMAConfig
from models.mol_llama_encoder import MolLLaMAEncoder
from transformers import AutoTokenizer, LlamaForCausalLM, PreTrainedModel, GenerationMixin
from unicore.data import Dictionary

from collections import defaultdict
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from openbabel import pybel
from torch_geometric.data import Data, Batch
from data_provider.mol_dataset import smiles2graph, get_unimol_data
from data_provider.collaters import Mol3DCollater
import numpy as np

def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self

class MolLLaMAPreTrainedModel(PreTrainedModel):
    config_class = MolLLaMAConfig
    base_model_prefix = 'mllm'
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"encoder.graph_encoder",
        r"llm."
    ]

class MolLLaMA(MolLLaMAPreTrainedModel):
    def __init__(
        self,
        config: MolLLaMAConfig,
        vocab_size=None,
        torch_dtype="float16",
        enable_flash=True,
    ):
        super().__init__(config)

        ## Intialize encoder
        self.encoder = MolLLaMAEncoder(
            graph_encoder_config = config.graph_encoder_config,
            blending_module_config = config.blending_module_config,
            qformer_config = config.qformer_config,
        )
        self.postprocess_encoder()

        ## Initialize LLM
        if torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32

        if enable_flash:
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config.llm_model, torch_dtype=torch_dtype, 
                                                            attn_implementation="flash_attention_2")
        else:
            self.llm = LlamaForCausalLM.from_pretrained(config.llm_config.llm_model, torch_dtype=torch_dtype)
        self.llm.resize_token_embeddings(vocab_size)
        
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                                    inference_mode=False,
                                    r=config.llm_config.lora_config.r,
                                    lora_alpha=config.llm_config.lora_config.lora_alpha,
                                    lora_dropout=config.llm_config.lora_config.lora_dropout,
                                    target_modules=['k_proj', 'v_proj', 'q_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'])
        self.peft_config = peft_config
        self.llm = get_peft_model(self.llm, peft_config)
        self.llm.print_trainable_parameters()
        
        self.llm_proj = nn.Linear(self.encoder.Qformer.config.hidden_size, 
                                    self.llm.config.hidden_size)

    def postprocess_encoder(self):
        self.encoder.Qformer.cls = None
        self.encoder.Qformer.bert.embeddings.word_embeddings = None
        self.encoder.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.encoder.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        self.encoder.graph_proj = None
        self.encoder.text_proj = None
        self.encoder.gtm_head = None

    def forward(self, graph_batch, text_batch):
        _, _, query_output = self.encoder.graph_forward(graph_batch)      
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
        inputs_embeds[text_batch.mol_token_flag] = \
            query_output.flatten(0, 1) # [batch_size, max_len, dim]

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            return_dict=True,
            labels=text_batch.labels,
            use_cache=False,
        )
        
        return outputs

    @torch.no_grad()
    def generate(
        self,
        graph_batch,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        _, _, query_output = self.encoder.graph_forward(graph_batch)
        query_output = self.llm_proj(query_output.last_hidden_state) #[batch_size,num_query_token,dim]

        inputs_embeds = self.llm.get_input_embeddings()(text_batch.input_ids) # [batch_size, max_len, dim]
        
        inputs_embeds[text_batch.mol_token_flag] = \
            query_output.flatten(0, 1).to(inputs_embeds.dtype) # [batch_size, max_len, dim]

        outputs = self.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=text_batch.attention_mask,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            top_p=top_p,
        )

        return outputs

    @torch.no_grad()
    def generate_with_smiles(
        self,
        smiles_list,
        text_batch,
        do_sample=False,
        num_beams=1,
        max_length=None,
        min_length=1,
        max_new_tokens=1024,
        min_new_tokens=None,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_return_sequences=1,
        top_p=None,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        graph_batch = get_mol_graphs(smiles_list, self.encoder.unimol_dictionary, self.device)
        outputs = self.generate(
            graph_batch=graph_batch,
            text_batch=text_batch,
            do_sample=do_sample,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id
        )
        return outputs

    def load_from_ckpt(self, ckpt_path):
        print(f"Loading from checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = {k[10:]:v for k,v in ckpt['state_dict'].items() if k.startswith("mol_llama.")}
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        for k in missing_keys:
            if 'position_ids' in k: continue
            assert k.startswith("encoder.graph_encoder.") or \
                    k.startswith("llm.")
        
    
    def load_from_stage1_ckpt(self, ckpt_path):
        print(f"Loading from stage1 checkpoint: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = {k[8:]:v for k,v in ckpt['state_dict'].items() if k.startswith("encoder.")}
        missing_keys, unexpected_keys = self.encoder.load_state_dict(state_dict, strict=False)
        
        assert len(unexpected_keys) == 0, f"unexpected keys: {unexpected_keys}"
        for k in missing_keys:
            assert k.startswith("graph_encoder.")
        
def gen_3d_conformation_from_rdkit(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        num_atoms = mol.GetNumAtoms()

        mol = Chem.AddHs(mol)
        AllChem.EmbedMultipleConfs(mol, numConfs=1, numThreads=8, pruneRmsThresh=1, maxAttempts=10000, useRandomCoords=False)
        try:
            AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=8)
        except:
            pass
        mol = Chem.RemoveHs(mol)
    except:
        return None, None
    if mol.GetNumConformers() == 0:
        return None, None

    if num_atoms != mol.GetNumAtoms():
        return None, None

    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = np.array(mol.GetConformer().GetPositions())
    return atoms, coordinates


def gen_3d_conformation_from_openbabel(smiles):
    mol = pybel.readstring('smi', smiles)
    mol.make3D(forcefield='mmff94', steps=10000)
    mol.OBMol.DeleteHydrogens()

    atomic_nums = [atom.atomicnum for atom in mol.atoms]
    pt = Chem.GetPeriodicTable()
    atoms = [pt.GetElementSymbol(atomic_num) for atomic_num in atomic_nums]
    coordinates = np.array([atom.coords for atom in mol.atoms])
    return atoms, coordinates


def gen_3d_conformation_from_libraries(smiles):
    atoms, coordinates = gen_3d_conformation_from_rdkit(smiles)
    if atoms is None or coordinates is None:
        atoms, coordinates = gen_3d_conformation_from_openbabel(smiles)

    return atoms, coordinates


def get_mol_graphs(smiles_list, dictionary, device):
    data_graphs = defaultdict(list)
    for idx, smiles in enumerate(tqdm(smiles_list, desc='Processing Molecules...')):
        atoms, coordinates = gen_3d_conformation_from_libraries(smiles)

        if atoms is None or coordinates is None:
            print(f"Invalid SMILES for {idx}-th SMILES: {smiles}")
            continue

        data_graphs['unimol'].append(
            get_unimol_data(atoms, coordinates, dictionary, remove_Hs=True))

        graph = smiles2graph(smiles)
        data_graphs['moleculestm'].append(Data(x=graph['node_feat'], 
                                        edge_index=graph['edge_index'], 
                                        edge_attr=graph['edge_feat']))

    d3_collater = Mol3DCollater(dictionary.pad())
    graph_batch = {}
    graph_batch['unimol'] = d3_collater(data_graphs['unimol'])
    graph_batch['moleculestm'] = Batch.from_data_list(data_graphs['moleculestm'])

    for key in graph_batch.keys():
        if key == 'unimol':
            for key_ in graph_batch[key].keys():
                graph_batch[key][key_] = graph_batch[key][key_].to(device)
        elif key == 'moleculestm':
            graph_batch[key] = graph_batch[key].to(device)
        
    return graph_batch