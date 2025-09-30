import os
from typing import Any, Dict
import json
import re
from collections import defaultdict

import torch
from torch import optim
import pytorch_lightning as pl
from torch_geometric.data import Batch

from trainer.optims import LinearWarmupCosineLRScheduler

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    
    ## try to print keys that are not included
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict


class MoleculeQATrainer(pl.LightningModule):
    def __init__(self, mol_llama, train_config):
        super().__init__()
        self.train_config = train_config

        self.mol_llama = mol_llama

        self.test_step_outputs = []

    def load_from_ckpt(self, ckpt_path):
        self.mol_llama.load_from_ckpt(ckpt_path)

    def configure_optimizers(self):
        self.trainer.fit_loop.setup_data()
        warmup_steps = min(len(self.trainer.train_dataloader), self.train_config.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.train_config.init_lr, weight_decay=self.train_config.weight_decay)
        if self.train_config.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.train_config.max_epochs, self.train_config.min_lr, self.train_config.init_lr, warmup_steps, self.train_config.warmup_lr)
        elif self.train_config.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        checkpoint.pop('optimizer_states')
        to_be_removed = []
        for key, value in checkpoint['state_dict'].items():
            try:
                if not self.get_parameter(key).requires_grad:
                    to_be_removed.append(key)
            except AttributeError:
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)

    def training_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = text_batch.input_ids.size(0)
        ###============== Overall Loss ===================###
        output = self.mol_llama(graph_batch, text_batch)
        loss = {'loss': output['loss']}

        self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
        return loss['loss']

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.mol_llama.generate(
            graph_batch, 
            text_batch,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
        pattern = r"[Aa]nswer:"

        # Generate further if the output does not contain "Answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nAnswer: ")
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = self.tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(self.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.device)

            new_responses = self.mol_llama.generate(
                new_graph_batch, 
                new_text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            )
            new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]



        for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
            self.test_step_outputs.append({
                'response': response,
                'answer': answer,
                'task': task
            })

        return responses

    def on_validation_epoch_end(self):
        outputs = self.test_step_outputs

        if len(self.train_config.devices) > 1:
            all_outputs = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [item for sublist in all_outputs for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, f"epoch{self.current_epoch}_test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graph_batch, text_batch, other_infos = batch
        responses = self.mol_llama.generate(
            graph_batch, 
            text_batch,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
        )
        generated_texts = self.tokenizer.batch_decode(responses, skip_special_tokens=True)
        original_texts = self.tokenizer.batch_decode(text_batch['input_ids'], skip_special_tokens=False)
        pattern = r"[Aa]nswer:"

        # Generate further if the output does not contain "Answer:"
        no_format_indices = []
        new_texts = []
        for idx, (original_text, generated_text) in enumerate(zip(original_texts, generated_texts)):
            if not re.search(pattern, generated_text):
                no_format_indices.append(idx)
                new_texts.append(original_text + generated_text + "\n\nAnswer: ")
        if len(no_format_indices) > 0:
            new_graph_batch = {"unimol": {}, "moleculestm": {}}
            new_text_batch = {}
            for k, v in graph_batch['unimol'].items():
                new_graph_batch['unimol'][k] = v[no_format_indices]
            new_graph_batch['moleculestm'] = Batch.from_data_list(graph_batch['moleculestm'].index_select(no_format_indices))

            new_text_batch = self.tokenizer(
                new_texts,
                truncation=False,
                padding="longest",
                return_tensors="pt",
                return_attention_mask=True,
                return_token_type_ids=False,
                add_special_tokens=False,
            ).to(self.device)
            new_text_batch.mol_token_flag = (new_text_batch.input_ids == self.tokenizer.mol_token_id).to(self.device)

            new_responses = self.mol_llama.generate(
                new_graph_batch, 
                new_text_batch,
                pad_token_id = self.tokenizer.pad_token_id,
                eos_token_id = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids('<|eot_id|>')],
            )
            new_generated_texts = self.tokenizer.batch_decode(new_responses, skip_special_tokens=True)

            for _, i in enumerate(no_format_indices):
                generated_texts[i] += "\n\nAnswer: " + new_generated_texts[_]



        for response, answer, task in zip(generated_texts, other_infos['answer'], other_infos['task']):
            self.test_step_outputs.append({
                'response': response,
                'answer': answer,
                'task': task
            })

        return responses

    def on_test_epoch_end(self):
        outputs = self.test_step_outputs

        if len(self.train_config.devices) > 1:
            all_outputs = [None for _ in range(self.trainer.world_size)]
            torch.distributed.all_gather_object(all_outputs, outputs)
            all_outputs = [item for sublist in all_outputs for item in sublist]
        else:
            all_outputs = outputs

        if self.global_rank == 0:
            corrects, results = self.compute_metrics(all_outputs)
            with open(os.path.join(self.logger.log_dir, "test_results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(self.logger.log_dir, "test_metrics.json"), "w") as f:
                json.dump(corrects, f, indent=4)
                

    def compute_metrics(self, outputs):
        results = defaultdict(list)
        corrects = {}

        for output in outputs:
            task = output['task']
            response = output['response']
            answer = output['answer'].replace("Answer: ", "")
            prediction = response.split("Answer: ")[-1].strip()
            if 'A' in prediction:
                prediction = 'A'
            elif 'B' in prediction:
                prediction = 'B'
            elif 'C' in prediction:
                prediction = 'C'
            elif 'D' in prediction:
                prediction = 'D'
            else:
                prediction = 'None'

            correct = 1 if prediction == answer else 0
            results[task].append({
                'response': response,
                'answer': answer,
                'prediction': prediction,
                'correct': correct
            })

            if task not in corrects:
                corrects[task] = {'correct': 0, 'total': 0}
            corrects[task]['correct'] += correct
            corrects[task]['total'] += 1

        tasks = list(corrects.keys())
        for task in tasks:
            correct = corrects[task]['correct']
            total = corrects[task]['total']
            accuracy = correct / total * 100
            corrects[task]['accuracy'] = accuracy

        # Calculate overall accuracy
        overall_correct = sum([corrects[task]['correct'] for task in tasks])
        overall_total = sum([corrects[task]['total'] for task in tasks])
        overall_accuracy = overall_correct / overall_total * 100
        corrects['overall'] = {'correct': overall_correct, 'total': overall_total, 'accuracy': overall_accuracy}

        return corrects, results

