import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import run_episode

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig


def main():
    with open("./eval_configs/01-random/maps.yaml", "r") as f:
        maps = yaml.safe_load(f)
    with open("./eval_configs/02-mazes/maps.yaml", "r") as f:
        maps2 = yaml.safe_load(f)
    maps.update(**maps2)
    env_cfg = Environment(
        with_animation=True,
        observation_type="MAPF",
        on_target="nothing",
        map=maps["validation-random-seed-001"],
        max_episode_steps=128,
        num_agents=32,
        seed=8,
        obs_radius=5,
        collision_system="soft",
    )

    env = create_eval_env(env_cfg)
    algo = MAPFGPTInference(
        MAPFGPTInferenceConfig(path_to_weights="weights/model-6M.pt")
    )
    algo.reset_states()
    results = run_episode(env, algo)
    env.save_animation(f"out.svg")
    print(results)


if __name__ == "__main__":
    main()
