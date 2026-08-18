# Robust-RL baseline for interval uncertainty in zeta

## What changed

The original pipeline learns robot paths first and applies the stochastic human model later during risk estimation. The robust baseline instead puts the human state inside the Bellman state:

```text
s = (robot_position, human_position, human_back_point)
```

`human_back_point` is included because it is used by the original `HumanAgent` movement rule.

The fixed human factor is replaced by

```text
zeta in [zeta_low, zeta_high]
```

For one precise zeta the Bellman target is

```text
Q_zeta(s,a) = E_zeta [ R + gamma max_a' Q(s',a') ]
```

For the robust baseline it becomes

```text
Q_R(s,a) = min_zeta E_zeta [ R + gamma max_a' Q_R(s',a') ]
```

and the policy chooses the largest robust action value:

```text
pi_R(s) = argmax_a Q_R(s,a)
```

This is the maximin step discussed in the meeting.

## Why only two zeta values are evaluated

This is not a two-sample approximation. In the original human rule each action probability is affine in `zeta`. For a fixed Bellman backup, the expected return is therefore affine in `zeta` as well. Its minimum on a closed interval is attained at an endpoint:

```text
min_{zeta in [L,U]} F(zeta) = min(F(L), F(U))
```

The continuous interval is therefore solved exactly for this one-dimensional model.


## Run

```bash
python main_robust.py --param yaml/10x10_robust.yaml
```

The default interval is `[0.07, 0.13]` around the original nominal value `0.10`. It is only a starting interval and should be justified experimentally before publication.

Run the tests with:

```bash
python -m unittest discover -s tests -v
```

## Results

Each run creates a timestamped directory in `results/robust_interval/` containing:

- `config.json`
- `metrics.csv`
- `initial_q_bounds.csv`
- `nominal_rollout.csv`
- `robust_rollout.csv`
- `nominal_path.png`
- `robust_path.png`
- `summary.txt`

`initial_q_bounds.csv` records the lower and upper Bellman value for each initial robot action and the induced `q_width`.
The interval width itself is `zeta_high - zeta_low`.

## Current scope

The baseline intentionally uses one human. With the back-point memory included, the exact joint state already grows considerably. 
The current implementation also assumes statewise (rectangular) interval uncertainty: the inner minimization over zeta is performed inside each Bellman backup.