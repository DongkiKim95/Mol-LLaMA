import torch
from transformers import BatchEncoding

def tokenize_messages_llama2(messages, tokenizer):
    tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for idx, m in enumerate(messages):
        if m['role'] == 'system':
            text = "<s>[INST] <<SYS>>\n" + m['content'] + "\n<</SYS>>\n\n"
            ignored = True

        elif m['role'] == 'user':
            if idx == 1:
                text = m['content'] + "[/INST]\n\n"
            else:
                text = "\n<s>[INST] " + m['content'] + " [/INST]\n\n"
            ignored = True

        elif m['role'] == 'assistant':  
            text = m['content'] + "</s>"
            ignored = False

        tokenized_ = tokenizer(text, add_special_tokens=False)
        tokenized["input_ids"].extend(tokenized_['input_ids'])
        tokenized["attention_mask"].extend(tokenized_['attention_mask'])
        if ignored:
            tokenized['labels'].extend([-100] * len(tokenized_['input_ids']))
        else:
            tokenized['labels'].extend(tokenized_['input_ids'])

    tokenized['mol_token_flag'] = [bool(t == tokenizer.mol_token_id) for t in tokenized['input_ids']]

    return BatchEncoding(data=tokenized)


def tokenize_messages_llama3(messages, tokenizer):
    tokenized = {
        "input_ids": [],
        "attention_mask": [],
        "labels": []
    }

    for m in messages:
        if m['role'] == 'system':
            text = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + \
                    m['content'] + "<|eot_id|>"
            ignored = True

        elif m['role'] == 'user':
            text = f"<|start_header_id|>user<|end_header_id|>\n\n" + m['content'] + "<|eot_id|>"
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            ignored = True

        elif m['role'] == 'assistant':
            text = m['content'] + "<|eot_id|>"
            ignored = False

        tokenized_ = tokenizer(text, add_special_tokens=False)
        tokenized["input_ids"].extend(tokenized_['input_ids'])
        tokenized["attention_mask"].extend(tokenized_['attention_mask'])
        if ignored:
            tokenized['labels'].extend([-100] * len(tokenized_['input_ids']))
        else:
            tokenized['labels'].extend(tokenized_['input_ids'])

    tokenized['mol_token_flag'] = [bool(t == tokenizer.mol_token_id) for t in tokenized['input_ids']]

    return BatchEncoding(data=tokenized)

def tokenized_messages(messages, tokenizer, llama_type):
    if llama_type == 'llama2':
        return tokenize_messages_llama2(messages, tokenizer)
    elif llama_type == 'llama3':
        return tokenize_messages_llama3(messages, tokenizer)
    else:
        raise ValueError("Unsupported Llama type. Choose 'llama2' or 'llama3'.")

def batch_tokenize_messages_list(messages_list, tokenizer, llama_type, padding_side='left'):
    tokenized_list = []
    for messages in messages_list:
        tokenized = tokenized_messages(messages, tokenizer, llama_type)
        tokenized_list.append(tokenized)

    # Pad the tokenized messages
    max_length = max(len(t['input_ids']) for t in tokenized_list)
    for t in tokenized_list:
        padding_length = max_length - len(t['input_ids'])

        if padding_side == 'left':
            t['input_ids'] = [tokenizer.pad_token_id] * padding_length + t['input_ids']
            t['attention_mask'] = [0] * padding_length + t['attention_mask']
            t['labels'] = [-100] * padding_length + t['labels']
            t['mol_token_flag'] = [False] * padding_length + t['mol_token_flag']
        else:
            t['input_ids'] += [tokenizer.pad_token_id] * padding_length
            t['attention_mask'] += [0] * padding_length
            t['labels'] += [-100] * padding_length
            t['mol_token_flag'] += [False] * padding_length
        t['input_ids'] = torch.tensor(t['input_ids'])
        t['attention_mask'] = torch.tensor(t['attention_mask'])
        t['labels'] = torch.tensor(t['labels'])
        t['mol_token_flag'] = torch.tensor(t['mol_token_flag'])

    tokenized_list = {
        'input_ids': torch.stack([t['input_ids'] for t in tokenized_list]),
        'attention_mask': torch.stack([t['attention_mask'] for t in tokenized_list]),
        'labels': torch.stack([t['labels'] for t in tokenized_list]),
        'mol_token_flag': torch.stack([t['mol_token_flag'] for t in tokenized_list]),
    }

    return BatchEncoding(data=tokenized_list)