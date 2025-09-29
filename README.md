# Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model

[[Project Page](https://mol-llama.github.io/)] [[Paper](https://arxiv.org/abs/2502.13449)] [[Data](https://huggingface.co/datasets/DongkiKim/Mol-LLaMA-Instruct)] [[Mol-Llama-2-7b-chat Weights](https://huggingface.co/DongkiKim/Mol-Llama-2-7b-chat)] [[Mol-Llama-3.1-8B-Instruct Weights](https://huggingface.co/DongkiKim/Mol-Llama-3.1-8B-Instruct)]

## Why Mol-LLaMA?

Mol-LLaMA is designed to elucidate fundamental properties of molecules with explainability and reasoning ability.

- Instruction Dataset: [Mol-LLaMA-Instruct](https://huggingface.co/datasets/DongkiKim/Mol-LLaMA-Instruct)
  - Broad knowledge centered on molecules including structural, chemical, and biological features of molecules.
  - Three data types considering the causality among molecular features: Detailed Structural Descriptions, Structure-to-Feature Relationships, and Comprehensive Conversations.
- Model Architecture
    1) Molecular encoders: Pretrained 2D encoder ([MoleculeSTM](https://huggingface.co/chao1224/MoleculeSTM)) and 3D encoder ([Uni-Mol](https://huggingface.co/dptech/Uni-Mol-Models))
    2) Blending Module: Combining complementary information from 2D and 3D encoders via cross-attention
    3) Q-Former: Embed molecular representations into query tokens based on [SciBERT](https://huggingface.co/allenai/scibert_scivocab_uncased)
    4) LoRA: Adapters for fine-tuning LLMs

## Dependencies
Create an environment with python 3.10 and pytorch 2.4.1. Then, run the following commands to install requirements:
```
pip install -r requirements.txt

pip install flash-attn --no-build-isolation
conda install pyg -c pyg -y
conda install pytorch-scatter -c pyg -y
conda install openbabel -c conda-forge -y
```

## Dataset Preparation
To train Mol-LLaMA, please download the dataset to `data` directory from the following link: [Mol-LLaMA-Instruct](https://huggingface.co/datasets/DongkiKim/Mol-LLaMA-Instruct)

## Model Preparation
The pretrained models, including LLaMA, SciBERT, MoleculeSTM, and Uni-Mol, will be automatically downloaded via hugggingface. FYI, the related lines can be found in `models/mol_llama.py` and `models/mol_llama_encoder.py`.

## Training
To train the projectors including the blending module and Q-Former, please run the following command:
```
python stage1.py
```
The training and data configurations can be found in `configs/stage1/`. Checkpoints will be saved in `checkpoints/stage1` as default.

To instruction-tune, please run the following command:
```
python stage2.py
```
The training and data configurations can be found in `configs/stage2/`. Checkpoints will be saved in `checkpoints/stage2` as default. We use the checkpoint of *10th epoch*.


## Playground
The exemplar inference code is provided in `playground.py` which takes `playground_inputs.json` as inputs. To run the inference, please follow the command below:
```
python playground.py
```

## Evaluation on PAMPA
To reproduce the results on PAMPA task, please run the following command:
```
python -m pampa.inference
```

## Citation

If you find our work useful, please consider citing our work.
```
@misc{kim2025molllama,
    title={Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Model},
    author={Dongki Kim and Wonbin Lee and Sung Ju Hwang},
    year={2025},
    eprint={2502.13449},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
```
