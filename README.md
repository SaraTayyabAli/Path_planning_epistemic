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

$$
\zeta \in [\zeta_{\mathrm{low}}, \zeta_{\mathrm{high}}].
$$

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

The probability of each non-preferred move is

$$
P(\text{each non-preferred move}) = \zeta.
$$

The probability of each preferred move is

$$
P(\text{each preferred move})
=
\frac{1 - n_{\mathrm{other}}\zeta}
     {n_{\mathrm{preferred}}}.
$$

The implementation validates that probabilities remain non-negative. Consequently, `zeta` must be small enough that:

$$
1 - n_{\mathrm{other}}\zeta \geq 0.
$$

The human state additionally stores `back_point`, which reduces the priority of immediately returning to the cell from which the human arrived.

### Reward model

The robot receives:

$$
R =
\begin{cases}
R_{\mathrm{goal}}, & \text{if the robot enters its goal cell}, \\
R_{\mathrm{step}}, & \text{otherwise}.
\end{cases}
$$

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

$$
Q_{\zeta}(s,a) =
\sum_{h',b'} P_{\zeta}(h',b' \mid h,b)
\left[
R(s,a,s') + \gamma \max_{a'} Q_{\zeta}(s',a')
\right].
$$

The robot transition is deterministic; the expectation is over possible next human states.

### Robust interval planning

For an uncertainty interval `[zeta_low, zeta_high]`, the lower-envelope robust backup is:

$$
Q^L(s,a) =
\min_{\zeta \in \{\zeta_{\mathrm{low}}, \zeta_{\mathrm{high}}\}}
\mathbb{E}_{P_{\zeta}}
\left[
R + \gamma \max_{a'} Q^L(s',a')
\right].
$$

The robust policy is:

$$
\pi_{\mathrm{robust}}(s)
=
\arg\max_a Q^L(s,a).
$$

This is a **maximin decision rule**: the robot chooses the action with the best worst-case endpoint value.

The solver also computes an upper envelope:

$$
Q^U(s,a) =
\max_{\zeta \in \{\zeta_{\mathrm{low}}, \zeta_{\mathrm{high}}\}}
\mathbb{E}_{P_{\zeta}}
\left[
R + \gamma \max_{a'} Q^U(s',a')
\right].
$$

The interval width

$$
Q^U(s,a) - Q^L(s,a)
$$

quantifies long-horizon sensitivity to the modeled human uncertainty.


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

