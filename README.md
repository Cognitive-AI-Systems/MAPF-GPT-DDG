# Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning



<div align="center" dir="auto">

---
[![arXiv](https://img.shields.io/badge/arXiv-2506.23793-b31b1b.svg)](https://arxiv.org/abs/2506.23793)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/CognitiveAISystems/MAPF-GPT-DDG/blob/main/LICENSE)
[![Hugging Face](https://img.shields.io/badge/Weights-MAPF--GPT-blue?logo=huggingface)](https://huggingface.co/aandreychuk/MAPF-GPT/tree/main)
[![Hugging Face](https://img.shields.io/badge/Dataset-MAPF--GPT-blue?logo=huggingface)](https://huggingface.co/datasets/aandreychuk/MAPF-GPT/tree/main)
[![Hugging Face](https://img.shields.io/badge/Dataset-MAPF--GPT--DDG-blue?logo=huggingface)](https://huggingface.co/datasets/aandreychuk/MAPF-GPT-DDG/tree/main)
</div>

The repository is based on the repository of original [MAPF-GPT](https://github.com/CognitiveAISystems/MAPF-GPT). It consists of the following crucial parts:

- `example.py` - an example of code to run the MAPF-GPT-DDG approach.
- `benchmark.py` - a script that launches the evaluation of the MAPF-GPT-DDG model on the POGEMA benchmark set of maps.
- `download_dataset.py` - a script that downloads 1B training dataset and 1M validation one. The dataset is uploaded to Hugging Face.
- `train.py` - a script that launches the training of the MAPF-GPT-DDG model.
- `eval_configs` - a folder that contains configs from the POGEMA benchmark. Required by the `benchmark.py` script.
- `ckpt_configs` - a folder that contains configs used for validation of intermidiate checkpoints used in ablation study.

## Local Installation

For local experiments using the model, it’s recommended to use `uv`.

To install the dependencies with `uv`, run:

```
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r docker/requirements.txt
```

## Docker Installation

It's recommended to utilize Docker to build the environment compatible with MAPF-GPT code. The `docker` folder contains both `Dockerfile` and `requirements.txt` files to successfully build an appropriate container.

```
cd docker & sh build.sh
```

## Running an example

To test MAPF-GPT-DDG, you can simply run the `example.py` script. By default, it uses the MAPF-GPT-DDG-2M model, but this can be adjusted.  
Additionally, there is a list of optional arguments: `--map_name`, `--device`, `--num_agents`, `--seed`, `--max_episode_steps`, `--model`, `--show_map_names`. The `--map_name` argument allows you to select a map from those available in the `eval_configs` folder. To list all available maps, you can provide the `--show_map_names` option or look inside `eval_config` folder. Here are a few examples from each set: `validation-random-seed-000`, `validation-mazes-seed-000`, `wfi_warehouse`, `Berlin_1_256_00`, `puzzle-00`.  

It is recommended to use GPU-accelerated setups; however, smaller models can be run on a CPU. For Apple Silicon machines, it's recommended to use `--device mps`, which significantly speeds up inference.
By default MAPF-GPT-DDG-2M model is used. However, the code is compitable with original MAPF-GPT weights as well, as the architecture of the model and the input/output stayed unchanged.
Thus, you can additionally choose from `2M`, `6M`, and `85M` model sizes of the original MAPF-GPT, which will be automatically downloaded from Hugging Face. Be aware that the 85M model requires 1 GB of disk space.


Here is an example of running MAPF-GPT-2M on a maze map:
```
python example.py --map_name validation-mazes-seed-000 --model 2M-DDG --num_agents 32
```


Here is an example of running MAPF-GPT-85M on `wfi_warehouse` map:
```
python example.py --map_name wfi_warehouse --model 85M --num_agents 192
```

In addition to statistics about SoC, success rate, etc., you will also get an SVG file that animates the solution found by MAPF-GPT, which will be saved to the `svg/` folder.


## Running evaluation

You can run the `benchmark.py` script, which will run both MAPF-GPT-2M and MAPF-GPT-DDG-2M models on all the scenarios from the POGEMA benchmark.
You can also run the MAPF-GPT-85M model by setting `path_to_weights` to `hf_weights/model-85M.pt`. The weights for all models will be downloaded automatically.

```
python benchmark.py
```

The results will be stored in the `eval_configs` folder near the corresponding configs. They can also be logged into wandb. The tables with average success rates will be displayed directly in the console.
You can also find the results (raw data and scripts to build plots) presented in the paper in the [metrics](https://github.com/Cognitive-AI-Systems/MAPF-GPT-DDG/tree/metrics) branch.

## Dataset

To train MAPF-GPT-DDG, we utilized the 1B dataset generated to train MAPF-GPT. It can be downloaded from Hugging Face via `download_dataset.py` script. During training phase we generate additional data, detecting hard cases, that cannot be efficiently solved by MAPF-GPT. Solving these hard cases by expert and adding new observation-action pairs to the training dataset allows to boost the performance of MAPF-GPT. In contrast to 1B dataset, it cannot be preliminary generated/downloaded as it requires to run MAPF-GPT on the instances to detect hard cases for the current checkpoint. More details about the generation of additional data and its usage during training are provided in the paper.


## Running training of MAPF-GPT with DDG

To train MAPF-GPT from scratch or fine-tune the existing weights, you can use the `train.py` script. By providing it a config, you can adjust the parameters of the model and training setup. The script utilizes DDP, which allows training the model on multiple GPUs simultaneously. By adjusting the `nproc_per_node` value, you can choose the number of GPUs that are used for training.
Adjusting the value of `dagger_type` parameter you can choose the way of how the model is trained:
  - `standard` - default MAPF-GPT training setup is utilized, without any additional data collection phases
  - `ddg` - during training an addiitonal dataset will be collected following the logic of the proposed DDG method
  - `dagger` - during training an addiitonal dataset will be collected following the logic of the classic DAgger method
```
torchrun --standalone --nproc_per_node=1 train.py gpt/config-2M-DDG.py
```

## Citation:

```bibtex
@article{andreychuk2025advancing,
  title={Advancing Learnable Multi-Agent Pathfinding Solvers with Active Fine-Tuning},
  author={Anton Andreychuk and Konstantin Yakovlev and Aleksandr Panov and Alexey Skrynnik},
  journal={arXiv preprint arXiv:2506.23793},
  year={2025},
  url={https://arxiv.org/abs/2506.23793}
}
```
