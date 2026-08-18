import argparse
import csv
import datetime
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from environment import Env
from method.robust_mprl import RobustQIteration


ACTION_NAMES = {0: "up", 1: "down", 2: "left", 3: "right"}


def load_scenario(path, num_humans=1):
    with open(path, "r", encoding="utf-8") as handle:
        param = yaml.load(handle, Loader=yaml.FullLoader)

    if num_humans != 1:
        raise ValueError(
            "The robust Bellman baseline currently supports one human. "
            "Validate the K=1 case before extending the state space."
        )

    robot_start = tuple(param["agents"]["start"])
    robot_goal = tuple(param["agents"]["goal"])
    size = tuple(param["map"]["dimensions"])
    obstacles = set(param["map"]["obstacles"])
    human_start = tuple(param["humans"]["human1"]["start"])
    human_goal = tuple(param["humans"]["human1"]["goal"])

    return param, robot_start, robot_goal, size, obstacles, human_start, human_goal


def choose_next_human(planner, human, back_point, zeta, rng):
    return planner.human_model.sample(human, back_point, zeta, rng)


def evaluate_policy(
    planner,
    policy,
    robot_start,
    human_start,
    zeta_true,
    episodes,
    budget,
    seed,
):
    collisions = []
    successes = []
    rewards = []
    path_lengths = []
    example = None

    for episode in range(episodes):
        rng = random.Random(seed + episode)
        robot = robot_start
        human = human_start
        back_point = None
        robot_path = [robot]
        human_path = [human]
        episode_collisions = 0

        for _ in range(budget):
            if robot == planner.robot_goal:
                break

            state = (robot, human, back_point)
            action = policy[state]
            robot_next = planner.env.get_state_action_space(robot, action)
            human_next, back_next = choose_next_human(
                planner, human, back_point, zeta_true, rng
            )

            same_cell = robot_next == human_next
            edge_swap = robot_next == human and human_next == robot
            if same_cell or edge_swap:
                episode_collisions += 1

            robot = robot_next
            human = human_next
            back_point = back_next
            robot_path.append(robot)
            human_path.append(human)

        goal_reached = robot == planner.robot_goal
        success = int(goal_reached and episode_collisions == 0)

        # Same reporting convention used in the original evaluation code.
        reward = len(robot_path) * planner.step_reward
        if goal_reached:
            reward += planner.goal_reward
        reward += episode_collisions * planner.collision_penalty

        collisions.append(episode_collisions)
        successes.append(success)
        rewards.append(reward)
        path_lengths.append(len(robot_path))

        if example is None:
            example = {
                "robot_path": robot_path,
                "human_path": human_path,
                "collisions": episode_collisions,
                "success": success,
                "reward": reward,
            }

    return {
        "success_rate": float(np.mean(successes)),
        "success_std": float(np.std(successes)),
        "average_conflicts": float(np.mean(collisions)),
        "conflict_std": float(np.std(collisions)),
        "average_reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "average_path_length": float(np.mean(path_lengths)),
        "path_length_std": float(np.std(path_lengths)),
        "example": example,
    }


def save_metrics(path, rows):
    fields = [
        "method",
        "true_zeta",
        "success_rate",
        "success_std",
        "average_conflicts",
        "conflict_std",
        "average_reward",
        "reward_std",
        "average_path_length",
        "path_length_std",
        "planning_seconds",
        "iterations",
        "zeta_width",
    ]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_initial_bounds(
    path,
    planner,
    bounds,
    initial_state,
    robust_policy,
    robust_optimal_actions,
    robust_q,
):
    margin = planner.action_margin(robust_q, initial_state)
    optimal = set(robust_optimal_actions[initial_state])

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "action",
                "zeta_low_lower_backup",
                "zeta_high_lower_backup",
                "worst_case_zeta",
                "lower_q",
                "upper_q",
                "q_width",
                "endpoint_span",
                "robust_optimal",
                "tie_break_selected",
                "robust_action_margin",
            ]
        )

        for action in planner.legal_actions[initial_state[0]]:
            row = bounds[(initial_state, action)]
            writer.writerow(
                [
                    ACTION_NAMES[action],
                    row["zeta_low_lower_backup"],
                    row["zeta_high_lower_backup"],
                    row["worst_case_zeta"],
                    row["lower_q"],
                    row["upper_q"],
                    row["q_width"],
                    row["endpoint_span"],
                    int(action in optimal),
                    int(robust_policy[initial_state] == action),
                    margin,
                ]
            )


def save_policy_comparison(
    path, planner, nominal_q, nominal_policy, robust_q, robust_policy, tolerance=1e-10
):
    fields = [
        "robot",
        "human",
        "back_point",
        "nominal_selected",
        "robust_selected",
        "nominal_optimal_actions",
        "robust_optimal_actions",
        "same_optimal_set",
        "disjoint_optimal_sets",
    ]

    rows = []
    set_difference = 0
    disjoint = 0

    for state in nominal_policy:
        nominal_optimal = set(planner.optimal_actions(nominal_q, state, tolerance))
        robust_optimal = set(planner.optimal_actions(robust_q, state, tolerance))
        same_set = nominal_optimal == robust_optimal
        no_overlap = nominal_optimal.isdisjoint(robust_optimal)

        if not same_set:
            set_difference += 1
        if no_overlap:
            disjoint += 1

        rows.append(
            {
                "robot": state[0],
                "human": state[1],
                "back_point": state[2],
                "nominal_selected": ACTION_NAMES[nominal_policy[state]],
                "robust_selected": ACTION_NAMES[robust_policy[state]],
                "nominal_optimal_actions": "|".join(
                    ACTION_NAMES[a] for a in sorted(nominal_optimal)
                ),
                "robust_optimal_actions": "|".join(
                    ACTION_NAMES[a] for a in sorted(robust_optimal)
                ),
                "same_optimal_set": int(same_set),
                "disjoint_optimal_sets": int(no_overlap),
            }
        )

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    return {
        "states_compared": total,
        "optimal_set_difference_count": set_difference,
        "optimal_set_difference_rate": set_difference / total if total else 0.0,
        "disjoint_optimal_count": disjoint,
        "disjoint_optimal_rate": disjoint / total if total else 0.0,
    }

def save_rollout(path, example):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "robot_x", "robot_y", "human_x", "human_y"])
        for step, (robot, human) in enumerate(
            zip(example["robot_path"], example["human_path"])
        ):
            writer.writerow([step, robot[0], robot[1], human[0], human[1]])


def plot_rollout(path, env, robot_goal, human_goal, example, title):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.grid(True)

    for obstacle in env.obstacles:
        ax.add_patch(plt.Rectangle((obstacle[0] - 0.5, obstacle[1] - 0.5), 1, 1))

    robot_x = [p[0] for p in example["robot_path"]]
    robot_y = [p[1] for p in example["robot_path"]]
    human_x = [p[0] for p in example["human_path"]]
    human_y = [p[1] for p in example["human_path"]]

    ax.plot(robot_x, robot_y, marker="o", label="robot")
    ax.plot(human_x, human_y, marker="x", label="human")
    ax.scatter(*example["robot_path"][0], marker="s", label="robot start")
    ax.scatter(*robot_goal, marker="*", s=120, label="robot goal")
    ax.scatter(*example["human_path"][0], marker="^", label="human start")
    ax.scatter(*human_goal, marker="D", label="human goal")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param", default="yaml/10x10_robust.yaml")
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--zeta-low", type=float, default=None)
    parser.add_argument("--zeta-high", type=float, default=None)
    parser.add_argument("--zeta-nominal", type=float, default=None)
    parser.add_argument("--simulation", type=int, default=100)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", default="results/robust_interval")
    args = parser.parse_args()

    (
        param,
        robot_start,
        robot_goal,
        size,
        obstacles,
        human_start,
        human_goal,
    ) = load_scenario(args.param)

    robust_cfg = param.get("robust", {})
    zeta_low = args.zeta_low if args.zeta_low is not None else robust_cfg.get("zeta_low", 0.02)
    zeta_high = args.zeta_high if args.zeta_high is not None else robust_cfg.get("zeta_high", 0.18)
    zeta_nominal = (
        args.zeta_nominal
        if args.zeta_nominal is not None
        else robust_cfg.get("zeta_nominal", 0.10)
    )
    gamma = args.gamma if args.gamma is not None else robust_cfg.get("gamma", 0.9)
    budget = args.budget if args.budget is not None else robust_cfg.get("budget", 20)

    if not zeta_low <= zeta_nominal <= zeta_high:
        raise ValueError("zeta_nominal must lie inside [zeta_low, zeta_high]")

    env = Env(robot_start, robot_goal, obstacles, size)
    planner = RobustQIteration(
        env,
        robot_goal=robot_goal,
        human_goal=human_goal,
        gamma=gamma,
        collision_penalty=robust_cfg.get("collision_penalty", -2.0),
    )

    initial_state = (robot_start, human_start, None)
    print(f"map: {size[0]}x{size[1]}")
    print(f"joint states: {len(planner.states)}")
    print(f"zeta interval: [{zeta_low}, {zeta_high}]")

    nominal_start = time.perf_counter()
    nominal_q, nominal_policy, nominal_iterations = planner.solve_nominal(
        zeta_nominal,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    nominal_time = time.perf_counter() - nominal_start

    robust_start = time.perf_counter()
    (
        robust_q,
        upper_q,
        robust_policy,
        robust_optimal_actions,
        bounds,
        robust_iterations,
        upper_iterations,
    ) = planner.solve_robust(
        zeta_low,
        zeta_high,
        tolerance=args.tolerance,
        max_iterations=args.max_iterations,
    )
    robust_time = time.perf_counter() - robust_start

    initial_robust_optimal = [
        ACTION_NAMES[action] for action in robust_optimal_actions[initial_state]
    ]
    print(
        "initial action: nominal={}, robust tie-break={}".format(
            ACTION_NAMES[nominal_policy[initial_state]],
            ACTION_NAMES[robust_policy[initial_state]],
        )
    )
    print(f"robust-optimal initial actions: {initial_robust_optimal}")
    print(f"planning time: nominal={nominal_time:.3f}s, robust={robust_time:.3f}s")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "param": args.param,
        "map_size": list(size),
        "robot_start": list(robot_start),
        "robot_goal": list(robot_goal),
        "human_start": list(human_start),
        "human_goal": list(human_goal),
        "zeta_low": zeta_low,
        "zeta_high": zeta_high,
        "zeta_nominal": zeta_nominal,
        "zeta_width": zeta_high - zeta_low,
        "gamma": gamma,
        "budget": budget,
        "simulation": args.simulation,
        "uncertainty_assumption": "statewise rectangular interval",
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    save_initial_bounds(
        output_dir / "initial_q_bounds.csv",
        planner,
        bounds,
        initial_state,
        robust_policy,
        robust_optimal_actions,
        robust_q,
    )

    policy_stats = save_policy_comparison(
        output_dir / "policy_comparison.csv",
        planner,
        nominal_q,
        nominal_policy,
        robust_q,
        robust_policy,
    )

    metric_rows = []
    examples = {}
    true_zetas = sorted(set([zeta_low, zeta_nominal, zeta_high]))

    for zeta_true in true_zetas:
        for method, policy, planning_seconds, iterations in [
            ("nominal_human_aware", nominal_policy, nominal_time, nominal_iterations),
            ("robust_interval", robust_policy, robust_time, robust_iterations),
        ]:
            result = evaluate_policy(
                planner,
                policy,
                robot_start,
                human_start,
                zeta_true,
                episodes=args.simulation,
                budget=budget,
                seed=args.seed,
            )
            examples[(method, zeta_true)] = result["example"]
            metric_rows.append(
                {
                    "method": method,
                    "true_zeta": zeta_true,
                    "success_rate": result["success_rate"],
                    "success_std": result["success_std"],
                    "average_conflicts": result["average_conflicts"],
                    "conflict_std": result["conflict_std"],
                    "average_reward": result["average_reward"],
                    "reward_std": result["reward_std"],
                    "average_path_length": result["average_path_length"],
                    "path_length_std": result["path_length_std"],
                    "planning_seconds": planning_seconds,
                    "iterations": iterations,
                    "zeta_width": zeta_high - zeta_low,
                }
            )

    save_metrics(output_dir / "metrics.csv", metric_rows)

    nominal_example = examples[("nominal_human_aware", zeta_nominal)]
    robust_example = examples[("robust_interval", zeta_nominal)]
    save_rollout(output_dir / "nominal_rollout.csv", nominal_example)
    save_rollout(output_dir / "robust_rollout.csv", robust_example)
    plot_rollout(
        output_dir / "nominal_path.png",
        env,
        robot_goal,
        human_goal,
        nominal_example,
        f"Example trajectory under nominal policy, true zeta={zeta_nominal}",
    )
    plot_rollout(
        output_dir / "robust_path.png",
        env,
        robot_goal,
        human_goal,
        robust_example,
        f"Example trajectory under robust interval policy, zeta in [{zeta_low}, {zeta_high}]",
    )

    summary_lines = [
        "Robust interval Bellman baseline",
        "",
        f"zeta interval: [{zeta_low}, {zeta_high}]",
        f"zeta width: {zeta_high - zeta_low}",
        f"nominal zeta: {zeta_nominal}",
        f"nominal initial action: {ACTION_NAMES[nominal_policy[initial_state]]}",
        f"robust tie-break initial action: {ACTION_NAMES[robust_policy[initial_state]]}",
        f"robust-optimal initial actions: {initial_robust_optimal}",
        f"robust action margin at initial state: {planner.action_margin(robust_q, initial_state)}",
        f"nominal iterations: {nominal_iterations}",
        f"robust lower iterations: {robust_iterations}",
        f"robust upper iterations: {upper_iterations}",
        f"nominal planning seconds: {nominal_time:.6f}",
        f"robust planning seconds: {robust_time:.6f}",
        f"optimal action-set difference rate: {policy_stats['optimal_set_difference_rate']:.6f}",
        f"disjoint optimal-action rate: {policy_stats['disjoint_optimal_rate']:.6f}",
        "",
        "The robust backup is min over the zeta interval, followed by max over robot actions.",
        "For this human model the Bellman target is affine in zeta, so checking both endpoints is exact.",
        "The implementation uses a statewise rectangular uncertainty assumption.",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"results saved to: {output_dir}")


if __name__ == "__main__":
    main()
