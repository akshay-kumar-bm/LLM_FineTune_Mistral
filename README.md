# LLM_FineTune_Mistral

Instruction fine-tuning of Mistral-7B (and trial runs on Gemma-2B) with LoRA using TorchTune, on an English OpenAssistant (OASST1) dataset converted to OpenAI chat format. Notebooks target AMD MI300 / CUDA GPUs.

## What it does
The repo documents three workflows as Jupyter notebooks:
1. Convert OpenAssistant `oasst1` to an English, OpenAI-style `{"messages": [...]}` conversation dataset and validate a fine-tuned model against it.
2. Fine-tune with a custom dataset using TorchTune LoRA recipes (configs for Gemma-2B included).
3. Fine-tune with a pre-built dataset (Alpaca-style) using standard TorchTune recipes.

Published artefacts referenced by the author (Hugging Face): the fine-tuned model `Akshaykumarbm/OpenAssisted-English-Mistral-7b` and the processed dataset `Akshaykumarbm/oasst-english-openai-formate`. Base model: `mistralai/Mistral-7B-v0.1`; source data: `OpenAssistant/oasst1`.

## Structure
```
complete_fine_tuing_human_like_convestion/
  dataprepocesing.ipynb   # OASST1 -> English -> conversation trees -> OpenAI format -> train JSON
  Validation.ipynb        # same conversion for the validation split + model comparison
  *.json                  # processed train (about 35 MB) and validation (about 1.7 MB) data
fine_tuing_custem_dataset/
  Data_creating_expro.ipynb   # `tune download`, recipe listing
  fineTuing.ipynb             # LoRA run, checkpoints
  gemma2bconfig.yaml          # LoRA config (Gemma-2B)
  data/                       # tiny sample datasets
Fine_tuning_pre_buil_dataset/
  Gemma2B_finetuing.ipynb, config.yaml, custom_config.yaml, custom_eval_config.yaml, custom_generation_config.yaml
.Trash-0/                     # accidentally committed trash folder (old notebooks, zips, a 38 MB jsonl)
```

## Tech stack
Python, PyTorch (ROCm or CUDA), TorchTune, Hugging Face `transformers`, `datasets`, `accelerate`, `huggingface_hub`, Jupyter.

## Setup / run
```bash
pip install torch torchvision torchtune transformers datasets accelerate huggingface_hub
huggingface-cli login            # needed for gated models / pushing datasets
tune download mistralai/Mistral-7B-v0.1 --hf-token <your token>
tune run lora_finetune_single_device --config mistral/7B_lora_single_device
```
Then walk through the notebooks in the order: `dataprepocesing` -> `fineTuing` -> `Validation`. (Commands above are from the project's own README; the notebooks use a mix of Mistral and Gemma/Llama recipes.)

## Status and limitations
- Single-device LoRA only; no automated evaluation (the author's README lists METEOR/ROUGE/BLEU as not implemented); validation is a manual side-by-side comparison.
- No training loss curves or benchmark numbers are committed in the README.
- Notebook paths point to a Docker mount (`/shared-docker/...`).
- Repository hygiene: `.ipynb_checkpoints`, `.Trash-0` and large JSON files are committed.
