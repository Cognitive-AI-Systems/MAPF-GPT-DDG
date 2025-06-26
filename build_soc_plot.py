import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def fetch_json_files(folder_path="01-random"):
    """
    Fetch all JSON files from the specified folder.
    
    Args:
        folder_path (str): Path to the folder containing JSON files
        
    Returns:
        list: List of paths to JSON files
    """
    # Make sure the folder path exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' does not exist.")
        return []
    
    # Get all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    
    if not json_files:
        print(f"No JSON files found in '{folder_path}'.")
    else:
        print(f"Found {len(json_files)} JSON file(s) in '{folder_path}'.")
    
    return json_files

def load_json_data(json_files):
    """
    Load data from JSON files.
    
    Args:
        json_files (list): List of paths to JSON files
        
    Returns:
        list: List of loaded JSON data
    """
    data_list = []
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                # Extract the algorithm name from the filename
                algorithm_name = os.path.basename(file_path).replace('.json', '')
                
                # Ensure data is a list
                if not isinstance(data, list):
                    data = [data]
                
                # Process each item in the list
                for item in data:
                    if isinstance(item, dict) and "metrics" in item and "env_grid_search" in item:
                        # Make sure algorithm is included
                        if "algorithm" not in item:
                            item["algorithm"] = algorithm_name
                        data_list.append(item)
                
                print(f"Successfully loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
        if "warehouse" in file_path:
            data_list = fix_seed(data_list)
    return data_list

def process_data_for_plotting(data_list):
    """
    Process the loaded JSON data to extract metrics by algorithm and number of agents.
    
    Args:
        data_list (list): List of loaded JSON data
        
    Returns:
        dict: Processed data organized by algorithm and number of agents
    """
    # Create a nested dictionary to store metrics by algorithm and instance
    processed_data = defaultdict(lambda: defaultdict(dict))
    
    for data_item in data_list:
        algorithm = data_item.get("algorithm", "Unknown")
        env_grid_search = data_item.get("env_grid_search", {})
        metrics = data_item.get("metrics", {})
        
        # Create an instance ID by combining map_name, seed, and num_agents
        map_name = env_grid_search.get("map_name", "unknown")
        seed = env_grid_search.get("seed", 0)  # Default seed to 0 if not present
        num_agents = env_grid_search.get("num_agents", 0)
        instance_id = f"{map_name}_{seed}_{num_agents}"
        
        # Store both SoC and CSR values for this algorithm and instance
        if map_name == "wfi_warehouse" and num_agents < 192:
            continue
        if "SoC" in metrics:
            processed_data[algorithm]["SoC"][instance_id] = metrics["SoC"]
        
        if "CSR" in metrics:
            processed_data[algorithm]["CSR"][instance_id] = metrics["CSR"]
    
    return processed_data

def fix_seed(data_list):
    """
    Add seeds to EPH logs where they might be missing.
    
    The function assumes logs are ordered by number of agents (all 32-agent instances,
    then all 64-agent instances, etc.) and assigns seeds 0-127 sequentially within
    each group of instances with the same number of agents.
    
    Args:
        data_list (list): List of loaded JSON data
        
    Returns:
        list: Updated data list with seeds added to EPH logs
    """
    # Group data by algorithm and number of agents
    eph_data_by_agents = {}
    
    # First, identify EPH data and group by number of agents
    for i, data_item in enumerate(data_list):
        algorithm = data_item.get("algorithm", "Unknown")
        
        if algorithm == "eph_random" or algorithm == "EPH":
            env_grid_search = data_item.get("env_grid_search", {})
            num_agents = env_grid_search.get("num_agents", 0)
            
            if num_agents not in eph_data_by_agents:
                eph_data_by_agents[num_agents] = []
            
            eph_data_by_agents[num_agents].append((i, data_item))
    
    # Now assign seeds sequentially for each group of agents
    for num_agents, items in eph_data_by_agents.items():
        print(f"Fixing seeds for {len(items)} EPH instances with {num_agents} agents")
        
        for seed, (index, data_item) in enumerate(items):
            if seed < 128:  # Limit to seeds 0-127
                # Add or update the seed in env_grid_search
                data_item["env_grid_search"]["seed"] = seed
                data_list[index] = data_item
            else:
                print(f"Warning: More than 128 instances found for {num_agents} agents, not all will receive seeds")
                break
    
    print(f"Finished adding seeds to EPH data")
    return data_list

def create_combined_soc_plot(plot_name="soc_combined.pdf"):
    # Create figure with 1x4 subplots
    fig, axs = plt.subplots(1, 4, figsize=(28, 6))    
    # Set larger font sizes
    plt.rcParams.update({
        'font.size': 20,
        'axes.titlesize': 24,
        'axes.labelsize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 20,
        'legend.fontsize': 16
    })
    
    # Define datasets and their titles
    datasets = ['random', 'mazes', 'warehouse', 'movingai']
    titles = ['Random Maps', 'Mazes Maps', 'Warehouse', 'Cities Tiles']
    
    # Define colors and labels for legend
    colors = ['#11c1acff', '#d1a683ff', '#637b87ff', '#005960','#674ea7ff', '#c1433c']
    labels = ['DCC', 'EPH', 'SCRIMP', 'MAPF-GPT-2M', 'MAPF-GPT-85M', 'MAPF-GPT-DDG-2M']
    
    # Process each dataset
    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        # Set up the folder path
        if dataset == 'random':
            folder_path = "01-random"
        elif dataset == 'mazes':
            folder_path = "02-mazes"
        elif dataset == 'warehouse':
            folder_path = "03-warehouse"
        elif dataset == 'movingai':
            folder_path = "04-movingai"
        
        # Load and process data
        json_files = ['LaCAM.json', 'DCC.json', 'EPH.json', 'SCRIMP.json', 'MAPF-GPT-2M.json', 'MAPF-GPT-85M.json', 'MAPF-GPT-DDG-2M.json']
        data_list = load_json_data([os.path.join(folder_path, file) for file in json_files])
        processed_data = process_data_for_plotting(data_list)
        
        # Create subplot
        create_boxplot(processed_data, axs[idx], title, colors)
        
        # Try to add map image
        try:
            import matplotlib.image as mpimg
            map_img = mpimg.imread(f'{folder_path}/{dataset}.png')
            
            # Define position based on dataset
            if dataset == 'mazes':
                pos = [0.582, 0.55, 0.45, 0.45]
            elif dataset == 'random':
                pos = [0.57, 0.55, 0.45, 0.45]
            elif dataset == 'warehouse':
                pos = [0.6, 0.63, 0.4, 0.4]
            elif dataset == 'movingai':
                pos = [0.582, 0.55, 0.45, 0.45]
                
            map_ax = axs[idx].inset_axes(pos, frameon=True)
            map_ax.imshow(map_img)
            map_ax.set_xticks([])
            map_ax.set_yticks([])
            for spine in map_ax.spines.values():
                spine.set_visible(True)
                
        except Exception as e:
            print(f"Error loading map image for {dataset}: {e}")
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color) for color in colors]
    fig.legend(legend_elements, labels,
              loc='center',
              bbox_to_anchor=(0.5, 0.02),
              ncol=6,
              fontsize=24)
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    
    # Save and show the plot
    plt.savefig(plot_name, bbox_inches='tight', dpi=300)
    # plt.show()

def create_boxplot(processed_data, ax, title, colors, reference_algorithm="LaCAM", remove_outliers=True):
    """Create a single boxplot in the given axis"""
    # Calculate relative SoC values
    relative_soc_by_algorithm = defaultdict(list)
    reference_soc = processed_data[reference_algorithm].get("SoC", {})
    reference_csr = processed_data[reference_algorithm].get("CSR", {})
    
    for algorithm, algorithm_data in processed_data.items():
        if algorithm == reference_algorithm:
            continue
        
        algorithm_soc = algorithm_data.get("SoC", {})
        algorithm_csr = algorithm_data.get("CSR", {})
        
        for instance_id, soc in algorithm_soc.items():
            ref_solved = reference_csr.get(instance_id, 0) > 0
            if not ref_solved:
                continue
            
            if instance_id in reference_soc:
                ref_soc = reference_soc[instance_id]
                if ref_soc > 0:
                    relative_soc = soc / ref_soc
                    relative_soc_by_algorithm[algorithm].append(relative_soc)
    
    # Prepare data for plotting
    data_to_plot = []
    for algorithm in ['DCC', 'EPH', 'SCRIMP', 'MAPF-GPT-2M', 'MAPF-GPT-85M', 'MAPF-GPT-DDG-2M']:
        if algorithm in relative_soc_by_algorithm:
            data_to_plot.append(relative_soc_by_algorithm[algorithm])
    
    # Create boxplot
    box = ax.boxplot(data_to_plot, patch_artist=True, labels=[' '] * len(data_to_plot),
                    showfliers=not remove_outliers, widths=0.6)
    
    # Style the boxplot
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    
    for median in box['medians']:
        median.set_color('black')
        median.set_linewidth(2.5)
    
    for whisker in box['whiskers']:
        whisker.set_linewidth(2)
    for cap in box['caps']:
        cap.set_linewidth(2)
    
    # Remove x-axis ticks
    ax.tick_params(axis='x', length=0)
    
    # Set labels and title
    ax.set_xlabel('')
    ax.set_ylabel('SoC Ratio', fontsize=24)
    ax.set_title(title, fontsize=28)
    
    ax.tick_params(axis='y', labelsize=24)
    # Add horizontal dotted lines
    ax.grid(False)
    ymin, ymax = ax.get_ylim()
    y_ticks = ax.get_yticks()
    for y in y_ticks:
        if ymin <= y <= ymax:
            ax.axhline(y=y, color='black', linestyle='--', alpha=1.0, linewidth=1, zorder=0)

if __name__ == "__main__":
    create_combined_soc_plot()