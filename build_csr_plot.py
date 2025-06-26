import json
from pathlib import Path

from pogema_toolbox.evaluator import run_views
from pogema_toolbox.views.view_multi_plot import MultiPlotView
from pogema_toolbox.views.view_utils import check_seeds


def fix_for_consistency_check(results):
    for x in results:
        if 'seed' not in x['env_grid_search']:
            x['env_grid_search']['seed'] = 0
        if 'map_name' not in x['env_grid_search']:
            x['env_grid_search']['map_name'] = 'none'


def multi_plot_split(results, name):
    for x in results:
        x['env_grid_search']['split'] = name


def create_combined_csr_plot(plot_name='01-CSR.pdf'):
    results = []

    evaluation_config = MultiPlotView(
        type='multi-plot',
        x='num_agents',
        y='CSR',
        over='split',
        width=2.5,
        height=2.5,
        line_width=2,
        num_cols=4,
        legend_columns=6,
        use_log_scale_x=True,
        legend_font_size=10,
        font_size=9,
        remove_individual_titles=False,
        remove_individual_legends=True,
        sort_over=False,
        ticks=[8, 16, 32, 64],
        hue_order=['MAPF-GPT-85M', 'MAPF-GPT-2M', 'DCC', 'SCRIMP', 'EPH', 'MAPF-GPT-DDG-2M'],
        # hue_order=['MAPF-GPTv1-85M', 'MAPF-GPTv1-2M', 'DCC', 'SCRIMP', 'EPH', 'MAPF-GPTv2-2M'],
        palette=['#674ea7ff', '#005960', '#11c1acff', '#637b87ff', '#d1a683ff', '#c1433c'],
        legend_bbox_to_anchor=(0.5, -0.12),
        rename_fields={
            'CSR': 'Success Rate',
            'num_agents': 'Number of Agents',
            '01-random': 'Random Maps'
        }
    )

    folders = ['01-random', '02-mazes', '03-warehouse', '04-movingai', ]
    mapping = {folder: name for folder, name in zip(folders, ['Random Maps', 'Mazes Maps', 'Warehouse', 'Cities Tiles'])}
    for folder in folders:
        for file in Path(folder).glob('*.json'):
            with open(file, 'r') as f:
                loaded = json.load(f)

                fix_for_consistency_check(loaded)
                multi_plot_split(loaded, name=mapping[folder])

                results += loaded

    check_seeds(results)
    run_views(results, {"results_views": {plot_name: evaluation_config.dict()}}, eval_dir='.')


if __name__ == '__main__':
    create_combined_csr_plot()