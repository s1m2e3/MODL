# MODL — Multi-Objective Deep Learning

A gradient-descent-compatible optimizer for training under **several objectives at
once**, applied to neural optimal control. When two objectives' gradients conflict,
MODL solves a small constrained problem for the weights that combine them, instead
of summing them with hand-tuned coefficients — and it does so **per parameter
tensor**, not once for the whole model.

## The problem with weighted sums

The default way to train against multiple objectives is a weighted sum,
`L = sum_i lambda_i L_i`, with the `lambda` values tuned by hand. Two things go
wrong:

- **The weights are a guess and they are global.** One set of coefficients has to
  serve every parameter in the network and every point in training.
- **Conflict is invisible.** If `grad L_1 . grad L_2 < 0`, the two objectives are
  pulling against each other and the sum silently cancels part of both. A weighted
  sum has no way to notice, let alone respond.

## What MODL does

`modl.py` is the whole algorithm, in four functions:

**1. `collect_gradients(loss, model, subset)`** — takes a per-objective gradient
with respect to a named subset of parameters (`torch.autograd.grad` with
`retain_graph=True`), clipped to `[-1, 1]`. Gradients are kept *separate*, one
dictionary per objective, rather than accumulated into `.grad`.

**2. `get_coordinated_lambda(...)`** — the decision, made **per named parameter**.
For each tensor it gathers the non-zero gradients that objectives produced for it,
and tests every pair for conflict:

```
conflict  <=>  dot(grad_i.flatten(), grad_j.flatten()) < 0   for some pair (i, j)
```

- **No conflict** — the objectives agree on this tensor, so just average them.
- **Conflict** — call `optimized_gradient`.

This granularity is the point. Different parts of a network serve different
objectives, and a controller's value head can be in conflict with its policy head
while the shared encoder is not. A global weighting cannot express that.

**3. `optimized_gradient(list_of_gradients)`** — solves for the combination
weights. It searches for `x` minimising the norm of the aggregate gradient's
projections onto each individual objective's gradient, via `torchmin`'s
`minimize_constr`. The result is an aggregate descent direction that gives each
conflicting objective a share, rather than letting the largest-magnitude gradient
win by default.

**4. `gradient_descent_step(lambdas, model, optimizer)`** — writes the aggregated
directions into `param.grad` and hands off to a standard `torch.optim` optimizer.
Any PyTorch optimizer works unchanged; MODL replaces the *gradient*, not the
update rule.

## Applied to constrained optimal control

The application is neural MPC: an RNN emits a control sequence over a horizon and
is trained by unrolling the dynamics through the loss. Here the multi-objective
machinery earns its place, because tracking performance and constraint
satisfaction are genuinely in conflict — the fastest way to a target is usually
straight through a bound.

Constraints are converted into *smooth, gradient-compatible* bound functions on
the control rather than a penalty added after the fact. In the oscillator study,
`position_upper_bound`, `control_upper_bound` and friends return the value the
control may not exceed given the current state, and the residual
`u - bound_rhs` is what the constraint objective acts on — with a big-M relaxation
and a softmax over active constraints (`bigM = 10`, `steepness = 5`,
`softmax_temperature = 0.1`) so that "which constraint binds" is differentiable.
This is the same output-constraining idea as
[sumo-intersection-controller](https://github.com/s1m2e3/sumo-intersection-controller),
reached from the optimizer side.

| study | system | objectives |
|---|---|---|
| `oscillator/` | mass-spring-damper, `torchdiffeq` integration, 300-step horizon | state tracking + position bounds + control bounds (+ optional velocity sign) |
| `car_following/` | kinematic vehicle: position, heading, speed, steering | target tracking + acceleration, steering-rate, centrifugal, min-distance and car-following bounds — 8 constraints, symbolically derived |
| `main_cart_pole.py` | Gym CartPole | PPO policy gradient + value loss + entropy + state-trajectory and difference losses, each routed to its own sub-network |

The car-following study derives its dynamics, loss and all constraint gradients
**symbolically with SymPy** (`car_following/symbolic_differentiation.py`), then
wraps the lambdified expressions as `torch.nn.Module`s (`BaseSympyModule`,
`SympyFuncModule`, `SympyConsModule`) so they drop into the autograd graph. Second
derivatives are cached too (`outputs/d2jd2u_5`), which is what makes the
Wolfe line search in `car_following/utils.py` affordable.

`main_cart_pole.py` shows the per-tensor routing most clearly: each loss collects
gradients only for the sub-network it concerns (`subset=['rnn2.' + name ...]`), so
`get_coordinated_lambda` sees a different set of competing objectives on different
tensors of the same model.

## Layout

```
modl.py                  the algorithm: collect_gradients, optimized_gradient,
                         get_coordinated_lambda, gradient_descent_step
model.py                 SimpleNN, InterdependentResidualRNN
mpc/mpc_nn.py            ResidualRNNController, NStepResidualRNNController
utils/sequence_rl.py     trajectory losses, returns, PPO / value / entropy losses
environment_simulation.py  run_sim: rollout loop against a Gym env
main_cart_pole.py        the CartPole study
test_optimizer.py        the optimizer in isolation
other_test.py, test.py

oscillator/
  system_model.py        dynamics, cost, the smooth bound functions,
                         ConstrainedResidualRNNController
  train_clean.py         unconstrained baseline   -> outputs/model_clean.pth
  train_constrained.py   constrained             -> outputs/model_constrained.pth
  test_clean.py          -> outputs/test_clean.csv
  test_constrained.py    -> outputs/test_constrained.csv
  process_data.r         figures

car_following/
  symbolic_functions.py         dynamics, quadratic loss, penalty terms and the
                                8 constraints, as SymPy expressions
  symbolic_differentiation.py   differentiate them and cache the lambdified
                                functions and Jacobians to outputs/
  system_model.py               the SymPy-to-torch Module wrappers
  train_constrained.py          training, with a CLI
  test_constrained.py           -> outputs/test_constrained.csv
  utils.py                      CommandLineArgs, Wolfe line search, soft_clip
  process_data.r, plots.r       figures
  outputs/                      committed weights, Jacobians and results

post_processing.R        top-level figures
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# requirements.txt is incomplete -- these are also imported:
pip install torchdiffeq stable-baselines3 gymnasium dill sympy \
            matplotlib pandas
```

`pytorch-minimize` provides the `torchmin` module that `modl.optimized_gradient`
and `oscillator/system_model.py` depend on — it is the one non-obvious
requirement. `box2d`/`pygame`/`gym` are for the Gym studies; `kagglehub` is
vestigial.

Note that `modl.py` sets `torch.autograd.set_detect_anomaly(True)` at import,
which is a large slowdown. Comment it out for any long run.

## Reproducing the results

### Oscillator

Run from inside `oscillator/` — the scripts use `./outputs/` relative paths.

```bash
cd oscillator

python train_clean.py            # unconstrained baseline
                                 # -> outputs/model_clean.pth
python train_constrained.py      # constrained, via MODL
                                 # -> outputs/model_constrained.pth

python test_clean.py             # -> outputs/test_clean.csv
python test_constrained.py       # -> outputs/test_constrained.csv
```

Both training scripts sweep 30 random initial conditions x 100 epochs over a
300-step horizon. `train_clean.py` uses hidden dim 32 at lr 0.01;
`train_constrained.py` uses hidden dim 128 at lr 0.001 — the test scripts
instantiate 128, so they pair with the constrained checkpoint. The two CSVs are
the comparison: same system, same target, with and without the constraint
objectives. Committed checkpoints and CSVs mean the figures reproduce without
retraining.

### Car following

```bash
cd car_following

# STEP 1 (slow, once): derive and cache the symbolic gradients
python symbolic_differentiation.py     # -> outputs/obj.pt, obj_jac.pt,
                                       #    constr_jac.pt, system_dynamics.pt,
                                       #    djdu_5, d2jd2u_5

# STEP 2: train
python train_constrained.py --weights_name weights_rnn \
                            --hidden_dim 512 --state_dim 3 --output_dim 2 \
                            --n_trajectory_steps 10 --n_init_conditions 10 \
                            --trajectory_optimization_epochs 100 \
                            --n_steps_alignment 10 --step_size_alignment 0.1
                                       # -> outputs/<weights_name>.pt

# STEP 3: evaluate
python test_constrained.py             # -> outputs/test_constrained.csv
```

`python train_constrained.py --help` lists every option (see
`CommandLineArgs` in `car_following/utils.py`). The committed
`outputs/weights_rnn.pt` and `outputs/weights_rnn_guided.pt` are a trained
unguided/guided pair, and the cached symbolic artifacts are committed too — so
step 1 can be skipped unless the dynamics or constraints change.

### CartPole

```bash
python main_cart_pole.py
```

`dqn_cartpole.zip` is a stable-baselines3 DQN reference agent.

### Figures

The R scripts read the CSVs the Python scripts produce:

```r
install.packages(c("ggplot2", "dplyr", "jsonlite", "tidyr"))
```

**Each R script hard-codes an absolute `setwd()` that must be edited first:**

| script | line | path to change |
|---|---|---|
| `post_processing.R` | 19 | `C:/Users/samil/Documents/MODL/outputs` |
| `oscillator/process_data.r` | 18 | `C:/Users/samil/Documents/MODL/oscillator` |
| `car_following/process_data.r` | 18 | `C:/Users/samil/Documents/MODL/car_following/outputs` |
| `car_following/plots.r` | 20 | the same |

Point each at your clone, or delete the line and set the working directory when
you run the script.

## Reproducibility notes

- Initial conditions are drawn with `torch.empty(...).uniform_(...)` and no seed,
  so every training run differs. Seed `torch` and `random` at the top for a
  repeatable run.
- The oscillator's clean and constrained scripts use different hidden dimensions
  (32 vs 128) — a checkpoint from one will not load into the other.
- There are two distinct `utils`: the top-level `utils/` package
  (`utils.sequence_rl`) and `car_following/utils.py`. Which one resolves depends
  on the working directory, which is why the per-directory instructions above
  matter.

## Context

Doctoral-era work at the University of Arizona. The smooth, gradient-compatible
constraining of network outputs is the subject of Chapter 4 of the dissertation
(*Enforcing Constraints in Neural Networks*); the multi-objective optimizer itself
is not part of the dissertation. Listed under Projects as *Multi-Objective
Gradient-Based Optimization*.

Author: Samuel Cornejo (<samuelcornejo@arizona.edu>)
