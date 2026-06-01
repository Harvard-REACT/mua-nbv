# mua_nbv_prediction

Temporal Gaussian Process Regression (TGPR) target trajectory predictor with
a constant-velocity (CV) prior. Used by both the simulation and testbed
trajectory predictor nodes. **No ROS 2 dependencies** (JAX + NumPy only).

## Algorithm

The predictor maintains a fixed-lag sliding window of `L` measurements and
performs GP regression over the state trajectory using a Markov kernel induced
by the CV dynamics model:

```
dx/dt = [0 I; 0 0] x + [0; I] w(t),   w(t) ~ WN(0, q_c)
```

State vector: `x = [px, py, vx, vy]`

At each step:
1. Build the prior mean trajectory by rolling out the CV model from an initial state estimate
2. Assemble the lifted prior covariance `K = A @ Q_big @ A^T` from transition matrices and process noise
3. Compute the GP posterior via the information form: `H = K^{-1} + C^T R^{-1} C`
4. Extract the current state (last window element) as the marginal mean and covariance
5. Propagate one step ahead: `x_{k+1} = F x_k`, `P_{k+1} = F P_k F^T + Q`

## Module

### `tgpr.TGPR_CV`

Fixed-lag TGPR smoother over measurement-derived state observations
`[px, py, vx, vy]`, where `vx, vy` are derived from position history.

```python
from mua_nbv_prediction.tgpr import TGPR_CV

gp = TGPR_CV(dataset_history=10, q_c=1.0, dt=1.0)
gp.measurements = jnp.array(measurements)  # (L, 4)
x_next, K_next, x_last, K_last = gp.predict_one_step(dt=1.0)
```
