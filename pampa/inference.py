
import argparse
from tqdm import tqdm
import re
import numpy as np

from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from transformers import AutoTokenizer
from models.mol_llama import MolLLaMA

from pampa.dataset import PAMPADataset, PAMPACollater



def main(args):
    # Load model and tokenizer
    llama_version = 'llama3' if 'Llama-3' in args.pretrained_model_name_or_path else 'llama2'
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_name_or_path)
    tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
    tokenizer.padding_side = 'left'
    if llama_version == 'llama3':
        terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|eot_id|>')]
    elif llama_version == 'llama2':
        terminators = tokenizer.eos_token_id

    model = MolLLaMA.from_pretrained(args.pretrained_model_name_or_path, vocab_size=len(tokenizer)).to(args.device)

    dataset = PAMPADataset(json_path='pampa/data/pampa.json', 
                        split='test', prompt_type=args.prompt_type, 
                        unimol_dictionary=model.encoder.unimol_dictionary)

    collater = PAMPACollater(tokenizer, model.encoder.unimol_dictionary, llama_version)
    dataloader = DataLoader(dataset, batch_size=32, collate_fn=collater, shuffle=False)

    pattern = r"[Ff]inal [Aa]nswer:"

    responses, answers, smiles_list = [], [], []
    for graph_batch, text_batch, answer, smiles in tqdm(dataloader):
        for key in graph_batch.keys():
            if key == 'unimol':
                for key_ in graph_batch[key].keys():
                    graph_batch[key][key_] = graph_batch[key][key_].to(args.device)
            elif key == 'moleculestm':
                graph_batch[key] = graph_batch[key].to(args.device)
        text_batch = text_batch.to(args.device)

        # Generate
        outputs = model.generate(
            graph_batch = graph_batch,
            text_batch = text_batch,
            pad_token_id = tokenizer.pad_token_id,
            eos_token_id = terminators,
        )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        original_texts = tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)

        # Generate further if the output does not contain "Final answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nFinal answer: ")
        
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(args.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == tokenizer.mol_token_id).to(args.device)

            new_generated_texts = model.generate(
                graph_batch = new_graph_batch,
                text_batch = new_text_batch,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = terminators,
            )

            new_generated_texts = tokenizer.batch_decode(new_generated_texts, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nFinal answer: " + new_generated_texts[_]
            
        responses.extend(generated_texts)
        answers.extend(answer)
        smiles_list.extend(smiles)

    
    true_pattern = r'[Hh]igh [Pp]ermeability'
    false_pattern = r'[Ll]ow-to-[Mm]oderate [Pp]ermeability|[Mm]oderate [Pp]ermeability'
    labels, preds = [], []
    for response, answer in zip(responses, answers):
        label = 1 if answer == "High permeability" else 0

        response = response.split("Final answer: ")[-1].strip()
        if re.search(true_pattern, response): pred = 1
        elif re.search(false_pattern, response): pred = 0
        else: pred = None

        labels.append(label)
        preds.append(pred)


    # Save the results
    with open(f'pampa/results_{llama_version}_{args.prompt_type}.txt', 'w') as f:
        for response, answer, smiles, label, pred in zip(responses, answers, smiles_list, labels, preds):
            f.write(f"SMILES: {smiles}\n")
            f.write('-'*50 + "\n")
            f.write(f"Label: {label}\n")
            f.write(f"Prediction: {pred if pred is not None else 'None'}\n")
            f.write('-'*50 + "\n")
            f.write(f"Response: {response}\n")
            f.write('-'*50 + "\n")
            f.write(f"Answer: {answer}\n")
            f.write("="*50 + "\n")

    # Calculate accuracy
    preds, labels = np.array(preds), np.array(labels)
    mask = preds != None
    labels = labels[mask]
    preds = preds[mask]


    accuracy = (preds == labels).sum() / len(labels) * 100
    print(f'Accuracy: {accuracy:.2f}%')

    with open(f'pampa/accuracy_{llama_version}_{args.prompt_type}.txt', 'w') as f:
        f.write(f'Accuracy: {accuracy:.2f}%\n')            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model_name_or_path', type=str, default='DongkiKim/Mol-Llama-3.1-8B-Instruct')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--prompt_type', type=str, default='default', choices=['default', 'rationale', 'task_info'],)
    args = parser.parse_args()
    main(args)

