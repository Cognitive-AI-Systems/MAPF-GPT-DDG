import glob
import hashlib
import json
import multiprocessing as mp
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_logging_env
from lacam.inference import LacamInference, LacamInferenceConfig
from tokenizer.generate_observations import ObservationGenerator
from tokenizer.parameters import InputParameters

EXPERT_DATA_FOLDER = "LaCAM_data"
TEMP_FOLDER = "temp"
DATASET_FOLDER = "dataset"
CONFIGS = [
    "dataset_configs/10-medium-mazes/10-medium-mazes-part1.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part2.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part3.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part4.yaml",
    "dataset_configs/12-medium-random/12-medium-random-part1.yaml",
]


def tensor_to_hash(tensor):
    tensor_bytes = tensor.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()


def add_tensors_and_actions_to_hdf5(
    name,
    hdf5_file,
    dataset_name_tensors,
    dataset_name_actions,
    tensors,
    actions,
    known_hashes=None,
):
    new_tensors = []
    new_actions = []
    duplicates = 0
    if known_hashes is None:
        known_hashes = set()
    for tensor, action in zip(tensors, actions):
        tensor_hash = tensor_to_hash(tensor)

        if tensor_hash not in known_hashes:
            known_hashes.add(tensor_hash)
            new_tensors.append(tensor)
            new_actions.append(action)
        else:
            duplicates += 1
    if len(new_tensors) > 0:
        actions_made = [0 for i in range(6)]
        for action in new_actions:
            actions_made[action] += 1
        i = len(new_tensors) - 1
        discarded = 0
        while i >= 0:
            if new_actions[i] == 5:
                if (actions_made[0] + actions_made[5]) > len(new_tensors) // 5:
                    new_actions.pop(i)
                    new_tensors.pop(i)
                    actions_made[5] -= 1
                    discarded += 1
                else:
                    new_actions[i] = 0
            i -= 1
        print(name, discarded, duplicates, len(new_tensors), actions_made)
        new_tensors = np.array(new_tensors)
        new_actions = np.array(new_actions)

        if dataset_name_tensors not in hdf5_file:
            maxshape_tensors = (None,) + new_tensors.shape[1:]
            hdf5_file.create_dataset(
                dataset_name_tensors,
                data=new_tensors,
                maxshape=maxshape_tensors,
                compression="lzf",
            )

            maxshape_actions = (None,) + new_actions.shape[1:]
            hdf5_file.create_dataset(
                dataset_name_actions,
                data=new_actions,
                maxshape=maxshape_actions,
                compression="lzf",
            )
        else:
            current_shape_tensors = hdf5_file[dataset_name_tensors].shape
            new_shape_tensors = (
                current_shape_tensors[0] + new_tensors.shape[0],
            ) + current_shape_tensors[1:]
            hdf5_file[dataset_name_tensors].resize(new_shape_tensors)
            hdf5_file[dataset_name_tensors][-new_tensors.shape[0] :] = new_tensors

            current_shape_actions = hdf5_file[dataset_name_actions].shape
            new_shape_actions = (
                current_shape_actions[0] + new_actions.shape[0],
            ) + current_shape_actions[1:]
            hdf5_file[dataset_name_actions].resize(new_shape_actions)
            hdf5_file[dataset_name_actions][-new_actions.shape[0] :] = new_actions
    return known_hashes


def generate_part(map_name):
    if os.path.isfile(map_name[:-4] + "hdf5"):
        try:
            with h5py.File(map_name[:-4] + "hdf5", "r") as file:
                print(map_name, len(file["input_tensors"]), "file is ok, continue")
                return
        except Exception as _:
            print(map_name, "file is not ok, regenerate it!")
    print("processing map", map_name)
    cfg = InputParameters()
    with open(map_name, "r") as f:
        data = json.load(f)
    if "random" in map_name:
        maps = yaml.safe_load(open("dataset_configs/12-medium-random/maps.yaml", "r"))
    else:
        maps = yaml.safe_load(open("dataset_configs/10-medium-mazes/maps.yaml", "r"))
    generator = ObservationGenerator(maps, data, cfg)
    tensors, gt_actions = generator.generate_observations(0, len(data))
    with h5py.File(map_name[:-4] + "hdf5", "w") as hdf5_file:
        add_tensors_and_actions_to_hdf5(
            map_name[:-4], hdf5_file, "input_tensors", "gt_actions", tensors, gt_actions
        )


def split_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    data_per_map = {}
    for d in data:
        if d["env_grid_search"]["map_name"] not in data_per_map:
            data_per_map[d["env_grid_search"]["map_name"]] = []
        data_per_map[d["env_grid_search"]["map_name"]].append(d)

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    for k, v in data_per_map.items():
        with open(f"{TEMP_FOLDER}/{k}.json", "w") as f:
            json.dump(v, f)


def generate_maps_hdfs():
    small_files = glob.glob(f"{TEMP_FOLDER}/*.json")
    with mp.Pool(processes=64) as pool:
        pool.map(generate_part, small_files)


def process_small_hdf5_files(big_hdf5_filename, small_files_pattern):
    known_hashes = set()

    with h5py.File(big_hdf5_filename, "a") as big_hdf5_file:
        if "input_tensors" in big_hdf5_file:
            for k in range(0, len(big_hdf5_file["input_tensors"]), 1000):
                tensors_dataset = big_hdf5_file["input_tensors"][
                    k : min(k + 1000, len(big_hdf5_file["input_tensors"]))
                ]
                for tensor in tensors_dataset:
                    tensor_hash = tensor_to_hash(tensor)
                    known_hashes.add(tensor_hash)

        small_files = glob.glob(small_files_pattern)
        for small_file in small_files:
            with h5py.File(small_file, "r") as sf:
                tensors = sf["input_tensors"][:]
                actions = sf["gt_actions"][:]
                print(len(tensors), len(known_hashes), small_file)
                known_hashes = add_tensors_and_actions_to_hdf5(
                    big_hdf5_filename,
                    big_hdf5_file,
                    "input_tensors",
                    "gt_actions",
                    tensors,
                    actions,
                    known_hashes,
                )


def save_arrow_chunk(chunk_input_tensors, chunk_gt_actions, schema, file_path):
    input_tensors_col = pa.array(chunk_input_tensors.tolist(), type=pa.list_(pa.int8()))
    gt_actions_col = pa.array(chunk_gt_actions.astype(np.int8))

    table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)

    with open(file_path, "wb") as f:
        with ipc.new_stream(f, schema) as writer:
            writer.write(table)
    print(f"Saved {file_path}")


def calculate_elements_to_pick(file_paths, total_pick_count):
    file_elements = {}
    total_elements = 0
    for file_path in file_paths:
        with h5py.File(file_path, "r") as f:
            num_elements = len(f["input_tensors"])
        file_elements[file_path] = num_elements
        total_elements += num_elements
    if total_pick_count > total_elements:
        print(
            f"Warning! Files don't contain enough data to pick {total_pick_count} elements. Using {total_elements} elements instead"
        )
        total_pick_count = total_elements

    elements_to_pick = {}
    total_picked = 0
    for file_path, num_elements in file_elements.items():
        elements_to_pick[file_path] = int(
            num_elements * total_pick_count / total_elements
        )
        total_picked += elements_to_pick[file_path]

    while total_picked < total_pick_count:
        for file_path, _ in file_elements.items():
            if total_picked == total_pick_count:
                break
            elements_to_pick[file_path] += 1
            total_picked += 1

    return elements_to_pick


def process_and_save_chunk(
    file_chunk,
    chunk_index,
    output_folder,
    max_elements_random=1024 * 1024 * 2,
    max_elements_mazes=1024 * 1024 * 18,
    num_files=10,
):
    data = []
    random_files_chunk = []
    mazes_files_chunk = []
    for file_path in file_chunk:
        if "random" in file_path:
            random_files_chunk.append(file_path)
        else:
            mazes_files_chunk.append(file_path)
    random_to_pick = calculate_elements_to_pick(random_files_chunk, max_elements_random)
    mazes_to_pick = calculate_elements_to_pick(mazes_files_chunk, max_elements_mazes)
    for file_path in file_chunk:

        try:
            with h5py.File(file_path, "r") as file:
                if "input_tensors" in file and "gt_actions" in file:
                    if "random" in file_path:
                        end_index = random_to_pick[file_path]
                    else:
                        end_index = mazes_to_pick[file_path]
                    input_tensors = file["input_tensors"][:end_index]
                    gt_actions = file["gt_actions"][:end_index]

                    if input_tensors.shape[0] != gt_actions.shape[0]:
                        raise ValueError(
                            "Mismatch in number of elements between input_tensors and gt_actions"
                        )

                    num_elements = input_tensors.shape[0]
                    data.append((input_tensors, gt_actions))

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    input_tensors_data = np.concatenate([x[0] for x in data], axis=0)
    gt_actions_data = np.concatenate([x[1] for x in data], axis=0)

    num_elements = len(input_tensors_data)
    indices = np.arange(num_elements)
    np.random.shuffle(indices)
    input_tensors_data = input_tensors_data[indices]
    gt_actions_data = gt_actions_data[indices]

    schema = pa.schema(
        [
            ("input_tensors", pa.list_(pa.int8())),  # List of 256 int8 values
            ("gt_actions", pa.int8()),
        ]
    )

    chunk_size = len(input_tensors_data) // num_files
    file_paths = []

    for i in range(num_files):
        start = i * chunk_size
        end = None if i == num_files - 1 else (i + 1) * chunk_size
        chunk_input_tensors = input_tensors_data[start:end]
        chunk_gt_actions = gt_actions_data[start:end]

        file_path = os.path.join(output_folder, f"chunk_{chunk_index}_file_{i}.arrow")
        file_paths.append((chunk_input_tensors, chunk_gt_actions, schema, file_path))
    for file_args in file_paths:
        save_arrow_chunk(*file_args)


def run_expert():
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_logging_env, Environment)
    ToolboxRegistry.register_algorithm("LaCAM", LacamInference, LacamInferenceConfig)
    unique_paths = {os.path.dirname(path) for path in CONFIGS}
    maps = {}
    for path in unique_paths:
        with open(f"{path}/maps.yaml", "r") as f:
            folder_maps = yaml.safe_load(f)
            maps.update = {**folder_maps}
    ToolboxRegistry.register_maps(maps)
    for config in CONFIGS:
        with open(config, "r") as f:
            evaluation_config = yaml.safe_load(f)

        eval_dir = Path(EXPERT_DATA_FOLDER) / config[:-5]
        initialize_wandb(evaluation_config, eval_dir, False, EXPERT_DATA_FOLDER)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


def generate_chunks():
    all_files = glob.glob(os.path.join(TEMP_FOLDER, "*.hdf5"))

    random.shuffle(all_files)
    num_chunks = 50
    chunk_size = len(all_files) // num_chunks
    chunks = [
        all_files[i : i + chunk_size] for i in range(0, len(all_files), chunk_size)
    ]

    os.makedirs(DATASET_FOLDER, exist_ok=True)
    with mp.Pool(5) as pool:
        pool.starmap(
            process_and_save_chunk,
            [(chunk, i, DATASET_FOLDER) for i, chunk in enumerate(chunks)],
        )


def main():
    # Step 1: Run LaCAM to obtain expert data in json format.
    run_expert()

    # Step 2: Load one (or mutiple) big json file and split it (them) into small ones (1 map = 1 json).
    files = [f"{EXPERT_DATA_FOLDER}/{config[:-5]}/LaCAM.json" for config in CONFIGS]
    with mp.Pool() as pool:
        pool.map(split_json, files)

    # Step 3: Generate observations.
    # Remove duplicates and redundant wait actions.
    # They are stored in hdf5 files with compression to reduce memory usage.
    generate_maps_hdfs()

    # Step 4: Generate dataset with chunk files.
    # Current settings create 50 chunks with 10 files in each of them.
    # Each chunk contains 2*2^20 observations from random maps and 18*2^20 observations from mazes.
    # 1B dataset requires 258GB of disk space (even being stored in int8 format).
    # Around 200 Gb of additional space is required to store intermediate data, i.e. json and hdf5 files.
    generate_chunks()


if __name__ == "__main__":
    main()
