# Robust Human-Aware Path Planning Under Epistemic Uncertainty

This repository implements a tabular robust planning baseline for grid-based robot navigation around a stochastic, goal-directed human. The central purpose is to study how **epistemic uncertainty** about human motion affects robot decisions, safety, and long-horizon value.

The method compares:

- a **nominal human-aware policy**, planned for one assumed human-behavior parameter `zeta_nominal`; and
- a **robust interval policy**, planned against an uncertainty interval `[zeta_low, zeta_high]`.

The current implementation supports **one robot and one human** on a static two-dimensional grid map with obstacles. The robot has deterministic motion; the human follows a stochastic transition model whose uncertainty is controlled by `zeta`.

---

## Overview

The project represents a robot–human navigation problem as a joint-state Markov decision process (MDP). A planning state is:

```python
(robot_position, human_position, human_back_point)
```
where

- `robot_position` is the robot grid cell;
- `human_position` is the human grid cell; and
- `human_back_point` is the human's previous cell, used to model one-step directional memory and discourage immediate reversals.

The robot chooses one of four actions:

```text
0: up
1: down
2: left
3: right
```

The robot transition is deterministic:

```text
robot state + robot action -> one next robot state
```

The human transition is stochastic. Given the current human state, its previous position, goal, and `zeta`, the model returns a distribution of possible next human states.

```text
human state + back point + zeta -> probability distribution over next states
```

The planner accounts for:

- robot step cost;
- robot goal reward;
- same-cell collisions; and
- edge-swap collisions, where robot and human exchange positions in one time step.

---

## Research motivation

Human-aware robot navigation depends on assumptions about how people move. A nominal planner usually assumes one fixed human-transition model, but that assumption can be wrong in practice. A pedestrian may be more or less goal-directed, more likely to stop, turn back, or take a non-preferred move than the robot expects.

This repository models that model uncertainty using an interval:

\[
\zeta \in [\zeta_{\mathrm{low}}, \zeta_{\mathrm{high}}].
\]

Here, `zeta` is the probability assigned to each non-preferred human move. A lower `zeta` produces more goal-directed and predictable motion; a higher `zeta` assigns more probability to deviations such as staying, reversing, moving laterally, or moving away from the goal.

The robust planner selects robot actions using a lower-envelope, worst-case Bellman backup. It therefore seeks actions with strong guaranteed performance across the specified uncertainty interval.

---

## Method

### Human transition model

For a human at state `h` with previous state `b`, the legal next cells include:

- remaining in the current cell;
- moving to valid non-obstacle cardinal neighbors;
- excluding invalid cells outside the map.

The human model identifies the move or moves that minimize Manhattan distance to the human goal. These are preferred moves. Each non-preferred move receives probability `zeta`; the remaining probability is shared equally by preferred moves.

For `n_other` non-preferred moves and `n_preferred` preferred moves:

\[
P(\text{each non-preferred move}) = \zeta,
\]

\[
P(\text{each preferred move}) =
\frac{1 - n_{\mathrm{other}}\zeta}{n_{\mathrm{preferred}}}.
\]

The implementation validates that probabilities remain non-negative. Consequently, `zeta` must be small enough that:

\[
1 - n_{\mathrm{other}}\zeta \geq 0.
\]

The human state additionally stores `back_point`, which reduces the priority of immediately returning to the cell from which the human arrived.

### Reward model

The robot receives:

\[
R =
\begin{cases}
R_{\mathrm{goal}}, & \text{if the robot enters its goal cell}, \\
R_{\mathrm{step}}, & \text{otherwise}.
\end{cases}
\]

A collision penalty is added if either of the following occurs:

1. **Same-cell collision**: robot and human enter the same cell.
2. **Edge-swap collision**: robot moves into the human's current cell while the human moves into the robot's current cell.

The default values are:

```text
Discount factor gamma:     0.9
Step reward:              -0.1
Goal reward:              10.0
Collision penalty:        -2.0
```

These values should be treated as experimental parameters, not universal safety settings. If collision represents physical failure, use a larger negative penalty and consider making collision terminal in both the planner and evaluation code.

### Nominal Bellman planning

For a fixed parameter `zeta`, the nominal Q-value update is:

\[
Q_{\zeta}(s,a) =
\sum_{h',b'} P_{\zeta}(h',b'\mid h,b)
\left[
R(s,a,s') + \gamma \max_{a'} Q_{\zeta}(s',a')
\right].
\]

The robot transition is deterministic; the expectation is over possible next human states.

### Robust interval planning

For an uncertainty interval `[zeta_low, zeta_high]`, the lower-envelope robust backup is:

\[
Q^L(s,a) =
\min_{\zeta \in \{\zeta_{\mathrm{low}},\zeta_{\mathrm{high}}\}}
\mathbb{E}_{P_{\zeta}}
\left[
R + \gamma \max_{a'} Q^L(s',a')
\right].
\]

The robust policy is:

\[
\pi_{\mathrm{robust}}(s)
=
\arg\max_a Q^L(s,a).
\]

This is a maximin decision rule: the robot chooses the action with the best worst-case endpoint value.

The solver also computes an upper envelope:

\[
Q^U(s,a) =
\max_{\zeta \in \{\zeta_{\mathrm{low}},\zeta_{\mathrm{high}}\}}
\mathbb{E}_{P_{\zeta}}
\left[
R + \gamma \max_{a'} Q^U(s',a')
\right].
\]

The interval width

\[
Q^U(s,a) - Q^L(s,a)
\]

quantifies long-horizon sensitivity to the modeled human uncertainty.

> **Important uncertainty interpretation:** the current implementation uses a **statewise rectangular uncertainty set**. Since the minimum is applied at each Bellman backup, the adverse endpoint may effectively vary from state to state. This is more conservative than assuming one globally fixed but unknown `zeta` for a full episode.

---

## Repository structure

```text
.
├── environment.py
├── human_transition.py
├── robust_mprl.py
├── main_robust.py
├── yaml/
│   └── 10x10_robust.yaml
└── results/
    └── robust_interval/
```

### `environment.py`

Defines the static grid world.

Responsibilities:

- stores grid dimensions, obstacle cells, start and goal locations;
- creates the set of traversable states;
- defines deterministic robot movement for four cardinal actions;
- returns legal human neighbor states, including a stay action;
- provides Manhattan and Euclidean distance helpers;
- can export grid/path visualizations as PDF.

### `human_transition.py`

Defines `HumanTransitionModel`.

Responsibilities:

- caches legal human neighbors;
- computes the full stochastic human transition distribution;
- uses Manhattan distance to identify goal-directed preferred moves;
- incorporates `back_point` memory to reduce immediate reversals;
- samples one transition for rollout simulation.

Main interfaces:

```python
model.distribution(state, back_point, zeta)
model.sample(state, back_point, zeta, rng)
```

### `robust_mprl.py`

Defines `RobustQIteration`.

Responsibilities:

- constructs the joint robot–human–memory state space;
- precomputes legal robot actions;
- evaluates collision-aware rewards;
- runs nominal Q-iteration;
- runs lower and upper robust envelope Q-iteration;
- extracts greedy policies and near-tied optimal action sets;
- creates per-state-action uncertainty diagnostics.

Main interfaces:

```python
planner.solve_nominal(zeta)
planner.solve_robust(zeta_low, zeta_high)
```

### `main_robust.py`

Runs a complete experiment.

Responsibilities:

- loads a YAML scenario;
- builds the environment and planner;
- solves nominal and robust policies;
- evaluates policies using simulated human trajectories;
- compares policies across low, nominal, and high true `zeta` values;
- saves metrics, policy comparisons, bounds, example rollouts, and path plots.

---

## Installation

Create and activate a virtual environment if desired:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install numpy matplotlib pyyaml reportlab
```

The code has been developed for Python 3. Ensure the import in `main_robust.py` matches your directory structure.

If `robust_mprl.py` is stored in the same directory as `main_robust.py`, use:

```python
from robust_mprl import RobustQIteration
```

If it is stored in a package directory called `method`, use:

```python
from method.robust_mprl import RobustQIteration
```

---

## Scenario configuration

The default scenario path is:

```text
yaml/10x10_robust.yaml
```

A scenario should define robot, human, map, and robust-planning parameters. For example:

```yaml
agents:
  start: [0, 0]
  goal: [9, 9]

humans:
  human1:
    start: [5, 2]
    goal: [5, 9]

map:
  # Current Env implementation interprets dimensions as [height, width].
  dimensions: [10, 10]
  obstacles:
    - [3, 3]
    - [3, 4]
    - [3, 5]

robust:
  gamma: 0.9
  zeta_low: 0.02
  zeta_nominal: 0.10
  zeta_high: 0.18
  budget: 20
  step_reward: -0.1
  goal_reward: 10.0
  collision_penalty: -2.0
```

The current `main_robust.py` reads `gamma`, `zeta_low`, `zeta_nominal`, `zeta_high`, `budget`, and `collision_penalty` from YAML. To configure step and goal rewards from YAML, pass `step_reward` and `goal_reward` explicitly when constructing `RobustQIteration`.

The nominal value should satisfy:

\[
\zeta_{\mathrm{low}}
\leq
\zeta_{\mathrm{nominal}}
\leq
\zeta_{\mathrm{high}}.
\]

---

## Running an experiment

Run the default experiment:

```bash
python main_robust.py
```

Run a specific scenario:

```bash
python main_robust.py --param yaml/10x10_robust.yaml
```

Override interval parameters from the command line:

```bash
python main_robust.py \
  --param yaml/10x10_robust.yaml \
  --zeta-low 0.02 \
  --zeta-nominal 0.10 \
  --zeta-high 0.18
```

Run more evaluation rollouts:

```bash
python main_robust.py \
  --simulation 500 \
  --budget 30 \
  --seed 42
```

Useful command-line arguments:

| Argument | Default | Description |
|---|---:|---|
| `--param` | `yaml/10x10_robust.yaml` | YAML scenario file |
| `--gamma` | YAML/default | Discount-factor override |
| `--zeta-low` | YAML/default | Lower uncertainty endpoint |
| `--zeta-high` | YAML/default | Upper uncertainty endpoint |
| `--zeta-nominal` | YAML/default | Nominal human parameter |
| `--simulation` | `100` | Evaluation episodes per policy and true parameter |
| `--budget` | YAML/default | Maximum actions per episode |
| `--tolerance` | `1e-6` | Q-iteration convergence tolerance |
| `--max-iterations` | `500` | Maximum Q-iteration sweeps |
| `--seed` | `123` | Base random seed |
| `--output` | `results/robust_interval` | Parent results directory |

---

## Outputs

Each run creates a timestamped directory such as:

```text
results/robust_interval/20260820_120000/
```

The output directory contains:

| File | Description |
|---|---|
| `config.json` | Complete run configuration and uncertainty assumption |
| `metrics.csv` | Aggregate evaluation metrics for both policies at each true `zeta` |
| `initial_q_bounds.csv` | Robust endpoint backups, lower/upper Q-values, widths, and initial action diagnostics |
| `policy_comparison.csv` | Nominal and robust actions across the full joint state space |
| `nominal_rollout.csv` | Example robot/human trajectory for the nominal policy |
| `robust_rollout.csv` | Example robot/human trajectory for the robust policy |
| `nominal_path.png` | Example nominal-policy path plot |
| `robust_path.png` | Example robust-policy path plot |
| `summary.txt` | Concise run summary, planning time, iterations, and policy-difference statistics |

### Metrics

The main evaluation metrics are:

- **Success rate:** fraction of episodes in which the robot reaches its goal with zero recorded collisions;
- **Average conflicts:** mean count of same-cell or edge-swap collisions per episode;
- **Average reward:** rollout reward based on step, goal, and collision terms;
- **Average path length:** mean number of states in the recorded robot trajectory;
- **Planning time:** wall-clock time for nominal or robust solution;
- **Iterations:** number of Q-iteration sweeps until convergence.

---

## Interpreting results

### Same nominal and robust path

It is possible—and often correct—for the nominal and robust policies to select the same robot path. This happens when uncertainty changes Q-values but does not change the ranking of feasible robot actions.

For example, both policies may prefer the direct route if it remains better than every detour even under the adverse endpoint.

In that case, the robust result should be interpreted as a **policy-stability certificate** over the chosen uncertainty interval, rather than as a route-improvement result.

To inspect whether robustness changes decisions, check:

```text
policy_comparison.csv
initial_q_bounds.csv
summary.txt
```

Particularly useful quantities are:

- `q_width`: long-horizon uncertainty sensitivity for a state-action pair;
- `endpoint_span`: one-step difference between endpoint backups;
- `robust_action_margin`: separation between the best and second-best robust actions;
- `optimal_set_difference_rate`: fraction of joint states where nominal and robust optimal action sets differ;
- `disjoint_optimal_rate`: fraction of states where nominal and robust optimal action sets have no action in common.

A visually identical single rollout does not prove two policies are identical across all states or stochastic human trajectories. For stronger analysis, compare actions only on states reached in many simulations under low, nominal, and high true `zeta` values.

### When robust planning should differ

Robust planning is most likely to select a different action when the map contains a genuine trade-off:

```text
Short route:  close to the human and uncertain collision exposure
Long route:   slightly longer but spatially safer
```

A narrow shared corridor, crossing bottleneck, or route close to the human's likely path is more informative than an open map where robot and human rarely interact.

---

## Limitations and current assumptions

- **One human only:** the joint state space currently supports one human. Multiple humans cause combinatorial state-space growth.
- **Full observability:** the robot policy receives the exact human position and `back_point`. This is not a POMDP or belief-space formulation.
- **Stationary policy:** after planning, the robot does not estimate or update `zeta` online.
- **Endpoint uncertainty:** the robust method evaluates `zeta_low` and `zeta_high`. The endpoint approach is valid for the current affine transition rule; it should be revalidated if the human model changes.
- **Statewise rectangular uncertainty:** the robust solver allows adverse endpoint selection locally in the Bellman recursion. It does not represent one fixed, unknown global `zeta` throughout a rollout.
- **Collision is non-terminal:** the default implementation penalizes collisions but permits the episode to continue. Choose terminal collision semantics if collision means physical safety failure.
- **Evaluation reward check:** evaluation should count one step reward per executed action. Since a stored path includes its initial position, use `len(robot_path) - 1` when reconstructing total step reward.
- **Computational cost:** robust planning is slower because it solves lower and upper envelope problems and evaluates two endpoint human models per robust backup.

---

## Suggested experiments

A useful experimental protocol is:

1. Define a human-aware bottleneck map with a short risky route and a longer safer detour.
2. Solve a nominal policy at `zeta_nominal`.
3. Solve an interval-robust policy for `[zeta_low, zeta_high]`.
4. Evaluate both policies under `zeta_low`, `zeta_nominal`, and `zeta_high`.
5. Use multiple random seeds and report mean and variation across rollouts.
6. Compare success rate, collisions, reward, path length, action disagreement on reachable states, Q-width, and planning time.
7. Perform sensitivity sweeps over interval width and collision penalty.

A robust planner is practically valuable when it either:

- selects a safer alternative action in uncertainty-sensitive states;
- improves worst-case collision-free success or reward; or
- demonstrates that a nominal policy remains stable across a formally specified uncertainty set.

---

## Future work

Possible extensions include:

- multiple humans with factored or approximate state representations;
- online estimation of human behavior parameters;
- belief-state planning under partial observability;
- learned human motion models from trajectories;
- nonlinear or multi-parameter uncertainty sets;
- chance-constrained collision-risk objectives;
- CVaR or distributionally robust objectives;
- terminal collision modeling for strict safety tasks;
- vectorized NumPy implementation or approximate dynamic programming for larger maps;
- comparison against the original MPRL method and other nominal human-aware planning baselines.

---

## Citation

If you use this repository in academic work, cite the associated thesis, report, or publication for this project. Add a BibTeX entry here once the work is publicly available.

```bibtex
@misc{robust_human_aware_path_planning,
  title  = {Robust Human-Aware Path Planning Under Epistemic Uncertainty},
  author = {Sara Tayyab Ali},
  year   = {2026},
  note   = {Software repository}
}
```

---

## License

Add a license file appropriate to your intended use and distribution, for example MIT, Apache-2.0, or a research-only license.
