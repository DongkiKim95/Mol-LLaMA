import argparse
import json

import os
from transformers import AutoTokenizer
from models.mol_llama import MolLLaMA, get_mol_graphs_from_data

from data_provider.tokenization_utils import batch_tokenize_messages_list


def get_text_data(level, num_data, tokenizer, llama_version, mol_prompt='<mol><mol><mol><mol><mol><mol><mol><mol>'):
    messages_list = []
    system = "You are a helpful assistant specializing in chemistry and biology. "\
            "The instruction that describes a task is given, paired with molecules. "\
            "Provide a response that appropriately completes the request."
    if level == 'structural':
        user = 'Explain the structural features of the given molecule.\nMolecule <mol>.'
    elif level == 'chemical':
        user = 'Explain the chemical properties of the given molecule.\nMolecule <mol>.'
    elif level == 'biological':
        user = 'Explain the biological properties of the given molecule.\nMolecule <mol>.'
    else:
        raise NotImplementedError

    for idx in range(num_data):
        messages = []
        messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user.replace("<mol>", mol_prompt)})

        messages_list.append(messages)
    
    text_batch = batch_tokenize_messages_list(messages_list, tokenizer, llama_version)
    return text_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--enable_flash', type=eval, default=False)
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='DongkiKim/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--device', type=str, default=0)

    parser.add_argument('--data_path', type=str, default='data/pubchem_test_representative/test-representative.json')
    parser.add_argument('--result_path', type=str, default='pubchem_infer_outputs/Mol-LLaMA')
    parser.add_argument('--level', type=str, default='structural', choices=['structural', 'chemical', 'biological'])
    
    args = parser.parse_args()
    if args.device != 'cpu':
        args.device = f'cuda:{args.device}'

    if args.precision == 'bf16-mixed':
        torch_dtype = "bfloat16"
    elif args.precision == '16':
        torch_dtype = "float16"
    elif args.precision == '32':
        torch_dtype = "float32"
    else:
        raise ValueError("Invalid precision type. Choose from 'bf16-mixed', '16', or '32'.")

    # Load model and tokenizer
    llama_version = 'llama3' if 'Llama-3' in args.pretrained_model_name_or_path else 'llama2'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    
    model = MolLLaMA.from_pretrained(args.pretrained_model_name_or_path, vocab_size=len(tokenizer),
                                    torch_dtype=torch_dtype, enable_flash=args.enable_flash).to(args.device)

    # Load inputs
    input_list = json.load(open(args.data_path, 'r'))
    for mol_data in input_list:
        mol_data['coordinates'] = mol_data['coordinates'][0]

    text_batch = get_text_data(args.level, len(input_list), tokenizer, llama_version).to(args.device)
    graph_batch = get_mol_graphs_from_data(input_list, model.encoder.unimol_dictionary, args.device)
    
    # Generate
    if llama_version == 'llama3':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif llama_version == 'llama2':
        terminators = tokenizer.eos_token_id

    outputs = model.generate(
        graph_batch = graph_batch,
        text_batch = text_batch,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = terminators
    )

    output_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    results = []
    for idx, out in enumerate(output_text):
        result = {
            'cid': input_list[idx]['cid'],
            'smiles': input_list[idx]['smiles'],
            'iupac_name': input_list[idx]['iupac_name'],
            'output': out,
            'description': input_list[idx]['description'],
        }
        results.append(result)

    os.makedirs(args.result_path, exist_ok=True)
    with open(f'{args.result_path}/{args.level}.json', 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Output saved to {args.result_path}/{args.level}.json")