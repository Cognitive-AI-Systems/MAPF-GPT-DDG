# MAPF-GPT: Imitation Learning for Multi-Agent Pathfinding at Scale

<div align="center" dir="auto">
   <p dir="auto"><img src="svg/puzzles.svg" alt="Follower" style="max-width: 100%;"></p>
</div>

---

The repository consists of the following crucial parts:

- `example.py` - an example of code to run the MAPF-GPT approach.
- `benchmark.py` - a script that launches the evaluation of the MAPF-GPT model on the POGEMA benchmark set of maps.
- `generate_dataset.py` - a script that generates a 1B training dataset. The details are provided inside the script in the main() function.
- `train.py` - a script that launches the training of the MAPF-GPT model.
- `weights` - a folder that contains pretrained models MAPF-GPT-2M and MAPF-GPT-6M. They are utilized in `example.py` and `benchmark.py` scripts.
- `eval_configs` - a folder that contains configs from the POGEMA benchmark. Required by the `benchmark.py` script.
- `dataset_configs` - a folder that contains configs to generate training and validation datasets. Required by the `generate_dataset.py` script.

## Installation

It's recommended to utilize Docker to build the environment compatible with MAPF-GPT code. The `docker` folder contains both `Dockerfile` and `requirements.txt` files to successfully build an appropriate container.

```
cd docker & sh build.sh
```

## Running an example

To test the work of MAPF-GPT, you can simply run the `example.py` script. You can adjust the parameters, such as the number of agents or the name of the map. You can also switch between MAPF-GPT-2M and MAPF-GPT-6M.

```
python3 example.py
```

Besides statistics about SoC, success rate, etc., you will also get an SVG file that animates the solution found by MAPF-GPT (`out.svg`).

## Running evaluation

You can run the `benchmark.py` script, which will run both MAPF-GPT-2M and MAPF-GPT-6M models on all the scenarios from the POGEMA benchmark.

```
python3 benchmark.py
```

The results will be stored in the `eval_configs` folder near the corresponding configs. They can also be logged into wandb. The tables with average success rates will be displayed directly in the console.

## Generating dataset

Due to the very large size of the dataset, we are not able to upload it to the repository. However, we provide a script that can complete all the steps required to generate the dataset, including the instance generation process (via POGEMA), solving the instances via LaCAM, generating and filtering observations, shuffling the data, and saving the dataset into multiple `.arrow` files for further efficient in-memory operation.

```
python3 generate_dataset.py
```

Please note that the full training dataset for 1B observation-gt_action pairs requires 258 GB of disk space and additionally around 200 GB for intermediate files. It also requires a lot of time to solve all instances via LaCAM. By modifying the config files in `dataset_configs` (adjusting the number of seeds, reducing the number of maps), you can reduce the time and space required to generate the dataset (as well as its final size).

## Running training of MAPF-GPT

To train MAPF-GPT from scratch or to fine-tune the existing models on other datasets (if you occasionally have such ones), you can use the `train.py` script. By providing it a config, you can adjust the parameters of the model and training setup. The script utilizes DDP, which allows training the model on multiple GPUs simultaneously. By adjusting the `nproc_per_node` value, you can choose the number of GPUs that are used for training.

```
torchrun --standalone --nproc_per_node=1 train.py gpt/config-6M.py
```
