import argparse

import time
import copy
import json
import os
import numpy as np
from openai import OpenAI
import re
import random
from concurrent.futures import ProcessPoolExecutor

def run_func(args, idx):
    gpt_result_path = os.path.join('pubchem_infer_outputs', 'GPT-4o', f'{args.level}.json')
    gen_result_path = os.path.join(args.result_path, f"{args.level}.json")
    print(gpt_result_path)
    print(gen_result_path)
    assert gpt_result_path.endswith('.json')
    assert gen_result_path.endswith('.json')
    assert args.level in gpt_result_path

    # --------- Get Results ---------
    with open(gpt_result_path, 'r') as f:
        gpt_results = json.load(f)

    with open(gen_result_path, 'r') as f:
        gen_results = json.load(f)

    assert len(gpt_results) == len(gen_results)

    for gpt_result, gen_result in zip(gpt_results, gen_results):
        cid = gpt_result['cid']
        assert cid == gen_result['cid']

    # --------- Prompt ---------
    
    system_prompt = 'You are a helpful assistant specializing in chemistry and biology. Your task is to evaluate the performance of two AI assistants in responding to a user question about a molecular explanation.\n\n'
    system_prompt += 'For your reference, the SMILES notation, IUPAC name, and a description of the given molecule are provided.\n\n'
    system_prompt += "Evaluate each assistant's response based on the following criteria: helpfulness, relevance, accuracy, and level of detail. Rate each criterion on a scale of 1 to 10, where a higher score indicates better performance. Additionally, provide an overall score for each assistant's response on a scale of 1 to 10.\n\n"
    system_prompt += "First output the scores of each assistant in the following format:\n"
    system_prompt += "[Assistant n]\n"
    system_prompt += "- Helpfulness: {score}\n"
    system_prompt += "- Relevance: {score}\n"
    system_prompt += "- Accuracy: {score}\n"
    system_prompt += "- Level of detail: {score}\n"
    system_prompt += "- Overall: {score}\n\n"
    system_prompt += "In the subsequent line, please provide a comprehensive explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    
    user_prompt_format = "[Molecule Information]\n\n"
    user_prompt_format += "SMILES: {}\n"
    user_prompt_format += "IUPAC Name: {}\n"
    user_prompt_format += "Description: {}\n\n"

    user_prompt_format += "[Question]\n"

    if args.level == 'structural':
        user_prompt_format += 'Explain the structural features of the given molecule.'
    elif args.level == 'chemical':
        user_prompt_format += 'Explain the chemical properties of the given molecule.'
    elif args.level == 'biological':
        user_prompt_format += 'Explain the biological properties of the given molecule.'
    else:
        raise NotImplementedError
    user_prompt_format += '\n\n'

    user_prompt_format += '[Assistant 1]\n{}\n\n[End of Assistant 1]\n\n'
    user_prompt_format += '[Assistant 2]\n{}\n\n[End of Assistant 2]\n\n'

    os.makedirs(args.root, exist_ok=True)

    print('--' * 50)
    print(f"Evaluating {args.eval_model} on {args.result_path} at {args.level} level, IDX: {idx}")
    print('--' * 50)

    # --------- Process Inputs ---------
    batch_messages = []
    batch_orders = [] # Indicate the location of Model respones
    for gpt_result, gen_result in zip(gpt_results, gen_results):
        smiles = gen_result['smiles']
        iupac_name = gen_result['iupac_name']
        description = gen_result['description']
        gen_response = gen_result['output']
        cid = gen_result['cid']

        if 'I can\u2019t provide' in gen_response:
            print(f"Skip {cid}")
            print(gen_response)
            continue
            
        gpt_response = gpt_result['output']

        if random.random() < 0.5:
            response1, response2 = gpt_response, gen_response
            batch_orders.append(2) # GPT -> 1, Model -> 2
        else:
            response1, response2 = gen_response, gpt_response
            batch_orders.append(1) # Model -> 1, GPT -> 2
        

        user_prompt = user_prompt_format.format(smiles, iupac_name, description, response1, response2)
        batch_messages.append([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

    output_path = os.path.join(args.root, f"{args.eval_model}_{str(idx)}")
    os.makedirs(os.path.join(output_path), exist_ok=True)

    
    # --------- Base Config for GPT ---------

    if 'gpt' in args.eval_model:
        base_config = {
            "method": "POST",
            "url": "/v1/chat/completions", 
            "body": {
                "model": args.eval_model, 
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": None}
                ],
                "max_tokens": 2048,
            }
        }

        model = OpenAI()
    else:
        raise ValueError()


    # --------- Generate Responses ---------
    if 'gpt' in args.eval_model:
        filename = os.path.join(output_path, 'input.jsonl')

        with open(filename, 'w') as out_file:
            for idx_, messages in enumerate(batch_messages):
                user_prompt = messages[1]['content']

                base_config_ = copy.deepcopy(base_config)

                base_config_['custom_id'] = f"{idx_}"
                base_config_['body']['messages'][1]['content'] = user_prompt
                
                json.dump(base_config_, out_file)
                out_file.write('\n')

        batch_input_file = model.files.create(
            file=open(filename, "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        created_metadata = model.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"{filename}"
            }
        )

        wait_time = 0
        while True:
            retrieved_metadata = model.batches.retrieve(created_metadata.id)
            if retrieved_metadata.status == "completed":
                break
            elif retrieved_metadata.status == "failed":
                raise Exception("Batch processing failed.")
            
            print(f"Waiting for batch processing to complete... Status: {retrieved_metadata.status}: {wait_time} seconds elapsed")
            wait_time += 30
            time.sleep(30)

        output_file_id = retrieved_metadata.output_file_id

        file_response = model.files.content(output_file_id)
        output_texts = file_response.text

        responses = []
        for line in output_texts.splitlines():
            output = json.loads(line)
            response = output['response']['body']['choices'][0]['message']['content']
            custom_id = output['custom_id']
            responses.append((custom_id, response))

        responses.sort(key=lambda x: int(x[0]))
        responses = [r[1] for r in responses]

    else:
        raise NotImplementedError("Only GPT evaluation is implemented in this script.")

    results = [{'response': response, 'order': order} for response, order in zip(responses, batch_orders)]
    # --------- Save Results ---------
    output_file = os.path.join(output_path, 'output.json')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)


    # --------- Evaluate Results ---------
    eval_results = []
    helpfulness_pattern = r"\**[Hh]elpfulness\**\s*:\**\s*([0-9]+(?:\.[0-9]+)?)"
    relevance_pattern = r"\**[Rr]elevance\**\s*:\**\s*([0-9]+(?:\.[0-9]+)?)"
    accuracy_pattern = r"\**[Aa]ccuracy\**\s*:\**\s*([0-9]+(?:\.[0-9]+)?)"
    level_of_detail_pattern = r"\**[Ll]evel of detail\**\s*:\**\s*([0-9]+(?:\.[0-9]+)?)"
    overall_pattern = r"\**[Oo]verall\**\s*:\**\s*([0-9]+(?:\.[0-9]+)?)"

    for result in results:
        # Extract scores of Assistant 1 and Assistant 2
        order = result['order']
        scores = {}

        think_flag = False

        for line in result['response'].split('\n'):
            if "<think>" in line:
                think_flag = True
                continue

            if think_flag:
                if "</think>" in line:
                    think_flag = False
                continue

            if "[Assistant" in line:
                assistant_num = line.split()[1].strip('[]')
                assistant_name = "Model" if order == int(assistant_num) else "GPT"
                scores[assistant_name] = {}

                continue

            if line == '\n':
                continue

            helpfulness_match = re.search(helpfulness_pattern, line)
            relevance_match = re.search(relevance_pattern, line)
            accuracy_match = re.search(accuracy_pattern, line)
            level_of_detail_match = re.search(level_of_detail_pattern, line)
            overall_match = re.search(overall_pattern, line)

            if helpfulness_match:
                key = "helpfulness"
                value = helpfulness_match.group(1)
                scores[assistant_name][key] = float(value)

            elif relevance_match:
                key = "relevance"
                value = relevance_match.group(1)
                scores[assistant_name][key] = float(value)

            elif accuracy_match:
                key = "accuracy"
                value = accuracy_match.group(1)
                scores[assistant_name][key] = float(value)
                
            elif level_of_detail_match:
                key = "level_of_detail"
                value = level_of_detail_match.group(1)
                scores[assistant_name][key] = float(value)
            
            elif overall_match:
                key = "overall"
                value = overall_match.group(1)
                scores[assistant_name][key] = float(value)
                
            else:
                pass

        wrong_flag = False
        for name in ['Model', 'GPT']:
            if name not in scores:
                wrong_flag = True
                continue
            for key in ['helpfulness', 'relevance', 'accuracy', 'level_of_detail', 'overall']:
                if key not in scores[name]:
                    wrong_flag = True
                    
        if wrong_flag:
            print("Skipping this result due to missing scores.")
            print(result['response'])
            continue

        eval_results.append(scores)

    # Average relative scores
    avg_scores = {}
    for key in ['helpfulness', 'relevance', 'accuracy', 'level_of_detail', 'overall']:
        avg_scores[key] = np.mean([result['Model'][key] for result in eval_results]) / np.mean([result['GPT'][key] for result in eval_results])

    print("Average Relative Scores:")
    for key, value in avg_scores.items():
        print(f"{key.capitalize()}: {value:.2f}")

    final_eval_results = {
        'avg_scores': avg_scores,
        'eval_results': eval_results
    }

    with open(os.path.join(output_path, 'final_eval_results.json'), 'w') as f:
        json.dump(final_eval_results, f, indent=4)

    print(f"Evaluation results saved to {os.path.join(output_path, 'final_eval_results.json')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_model', type=str, choices=['gpt-4o-2024-08-06'], default='gpt-4o-2024-08-06')
    parser.add_argument('--result_path', type=str, default='pubchem_infer_outputs/Mol-LLaMA/')
    parser.add_argument('--level', type=str, choices=['structural', 'chemical', 'biological'], required=True)
    args = parser.parse_args()

    args.root = os.path.join(args.result_path, args.level)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_func, args, idx) for idx in range(3)]
        for future in futures:
            future.result()

    print("All tasks completed.")