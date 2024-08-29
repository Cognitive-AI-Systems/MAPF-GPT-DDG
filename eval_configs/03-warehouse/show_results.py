from pathlib import Path

from pogema_toolbox.evaluator import run_views
from pogema_toolbox.views.view_utils import load_from_folder, check_seeds


def main():
    current_dir_name = Path(__file__).parent

    results, evaluation_config = load_from_folder(current_dir_name)
    check_seeds(results)
    run_views(results, evaluation_config, eval_dir=current_dir_name)


if __name__ == '__main__':
    main()
