from transformers import PretrainedConfig

class LoraConfig(PretrainedConfig):
    model_type = 'mol_llama_lora'
    base_config_key = 'lora_config'

    def __init__(
        self,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout

class LLMConfig(PretrainedConfig):
    model_type = 'mol_llama_llm'
    base_config_key = 'llm_config'
    is_composition = True
    sub_configs = {"lora_config": LoraConfig}

    def __init__(
        self,
        llm_model='meta-llama/Llama-3.1-8B-Instruct',
        lora_config=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if lora_config is None:
            lora_config = {}

        self.lora_config = LoraConfig(**lora_config)
        self.llm_model = llm_model

class QformerConfig(PretrainedConfig):
    model_type = 'mol_llama_qformer'
    base_config_key = 'qformer_config'

    def __init__(
        self,
        bert_name='allenai/scibert_scivocab_uncased',
        num_query_tokens=8,
        cross_attention_freq=2,
        embed_dim=256,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.bert_name = bert_name
        self.num_query_tokens = num_query_tokens
        self.cross_attention_freq = cross_attention_freq
        self.embed_dim = embed_dim

class BlendingModuleConfig(PretrainedConfig):
    model_type = 'mol_llama_blending_module'
    base_config_key = 'blending_module_config'

    def __init__(
        self,
        num_layers=4,
        num_heads=8,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_heads = num_heads

class UniMolConfig(PretrainedConfig):
    model_type = 'mol_llama_unimol'
    base_config_key = 'unimol_config'

    def __init__(
        self,
        repo_id='dptech/Uni-Mol-Models',
        weights_filename='mol_pre_no_h_220816.pt',
        dictionary_filename='mol.dict.txt',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repo_id = repo_id
        self.weights_filename = weights_filename
        self.dictionary_filename = dictionary_filename

class MoleculeSTMConfig(PretrainedConfig):
    model_type = 'mol_llama_molecule_stm'
    base_config_key = 'moleculestm_config'

    def __init__(
        self,
        repo_id='chao1224/MoleculeSTM',
        filename='demo/demo_checkpoints_Graph/molecule_model.pth',
        **kwargs
    ):
        super().__init__(**kwargs)
        self.repo_id = repo_id
        self.filename = filename

class GraphEncoderConfig(PretrainedConfig):
    model_type = 'mol_llama_graph_encoder'
    base_config_key = 'graph_encoder_config'
    is_composition = True
    sub_configs = {"unimol_config": UniMolConfig, "moleculestm_config": MoleculeSTMConfig}

    def __init__(
        self,
        unimol_config=None,
        moleculestm_config=None,
        encoder_types=['unimol', 'moleculestm'],
        **kwargs
    ):
        super().__init__(**kwargs)
        if unimol_config is None:
            unimol_config = {}

        if moleculestm_config is None:
            moleculestm_config = {}
        self.unimol_config = UniMolConfig(**unimol_config)
        self.moleculestm_config = MoleculeSTMConfig(**moleculestm_config)
        self.encoder_types = encoder_types

class MolLLaMAConfig(PretrainedConfig):
    model_type = 'mol_llama'
    is_composition = True
    sub_configs = {"qformer_config": QformerConfig, "blending_module_config": BlendingModuleConfig, "graph_encoder_config": GraphEncoderConfig, "llm_config": LLMConfig}

    def __init__(
        self,
        qformer_config=None,
        blending_module_config=None,
        graph_encoder_config=None,
        llm_config=None,
        torch_dtype="float16",
        **kwargs
    ):
        super().__init__(**kwargs)
        if qformer_config is None:
            qformer_config = {}

        if blending_module_config is None:
            blending_module_config = {}

        if graph_encoder_config is None:
            graph_encoder_config = {}

        if llm_config is None:
            llm_config = {}

        self.qformer_config = QformerConfig(**qformer_config)
        self.blending_module_config = BlendingModuleConfig(**blending_module_config)
        self.graph_encoder_config = GraphEncoderConfig(**graph_encoder_config)
        self.llm_config = LLMConfig(**llm_config)