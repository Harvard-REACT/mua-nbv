"""
Temporal Gaussian Process Regression with a Constant-Velocity prior (TGPR-CV).

Fixed-lag GP smoother over a sliding window of measurement-derived state
observations. The Markov prior kernel is induced by the 2-D
constant-velocity SDE:

    dx/dt = [0 I; 0 0] x + [0; I] w(t),   w(t) ~ WN(0, q_c)

State vector and observation vector:

    x_k = z_k = [px, py, vx, vy]   (dim = 4)
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag, solve


class TGPR_CV:
    """Fixed-lag GP smoother with constant-velocity prior.

    Parameters
    ----------
    dataset_history : int
        Window size L (number of measurements kept).
    q_c : float
        Continuous-time process-noise power spectral density.
    K0 : jnp.ndarray | None
        Initial state covariance (4x4).  Defaults to
        diag([0.01, 0.01, 0.1, 0.1]).
    R : jnp.ndarray | None
        Per-step measurement noise (2x2) for position-only measurements ``[px, py]``.
    dt : float
        Nominal time step between measurements.
    observe_position_only : bool
        Kept for API compatibility; this implementation is position-only.
    """

    def __init__(
        self,
        dataset_history: int = 10,
        q_c: float = 1.0,
        K0: jnp.ndarray | None = None,
        R: jnp.ndarray | None = None,
        dt: float = 1.0,
        observe_position_only: bool = True,
    ):
        self._max_hist = int(dataset_history)
        self.observe_position_only = True
        self._meas_dim = 2
        self._measurements = jnp.empty((0, self._meas_dim))
        self.q_c = float(q_c)
        self.dt = float(dt)

        if K0 is None:
            K0 = jnp.diag(jnp.array([0.01, 0.01, 0.1, 0.1], dtype=jnp.float32))
        self.K0 = K0

        if R is None:
            R = jnp.diag(
                jnp.array([0.01, 0.01], dtype=jnp.float32)
            )
        self.R = R

        # Maps each 4D state block to position-only observations (Barfoot-style).
        C_single = jnp.concatenate(
            [jnp.eye(2, dtype=jnp.float32), jnp.zeros((2, 2), dtype=jnp.float32)],
            axis=1,
        )
        L = self._max_hist
        self.C_big = jnp.kron(jnp.eye(L, dtype=jnp.float32), C_single)
        self.R_inv = jnp.kron(
            jnp.eye(L, dtype=jnp.float32), jnp.linalg.inv(self.R)
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def meas_dim(self) -> int:
        return self._meas_dim

    @property
    def window_size(self) -> int:
        return self._max_hist

    @property
    def dataset_size(self) -> int:
        return int(self._measurements.shape[0])

    @property
    def measurements(self) -> jnp.ndarray:
        return self._measurements

    @measurements.setter
    def measurements(self, value: jnp.ndarray) -> None:
        if int(value.shape[-1]) != self._meas_dim:
            raise ValueError(
                f"TGPR_CV: expected measurements with last dim {self._meas_dim}, "
                f"got shape {value.shape}"
            )
        self._measurements = value

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_one_step(
        self, dt: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Run the fixed-lag smoother and propagate one step ahead.

        Returns
        -------
        x_next : (4,)   predicted state mean at t+dt
        K_next : (4,4)  predicted state covariance at t+dt
        x_last : (4,)   smoothed state mean at t (last window element)
        K_last : (4,4)  smoothed state covariance at t
        """
        if self.dataset_size != self._max_hist:
            raise ValueError(
                f"TGPR_CV expects dataset_size==L. "
                f"Got N={self.dataset_size}, L={self._max_hist}"
            )

        x0 = self._init_state_xyv(self._measurements)
        x_bar = self._prior_rollout_cv(x0, dt).reshape(-1)

        F = self._F_cv(dt)
        Q = self._Q_cv(dt)

        n_trans = self.dataset_size - 1
        Phi_list = jnp.tile(F[None, :, :], (n_trans, 1, 1))
        Q_list = jnp.tile(Q[None, :, :], (n_trans, 1, 1))

        A_lift = self._A_lift(Phi_list)
        Q_big = block_diag(self.K0, *Q_list)

        K = (
            A_lift @ Q_big @ A_lift.T
            + jnp.eye(A_lift.shape[0], dtype=jnp.float32) * 1e-8
        )
        K_inv = jnp.linalg.inv(K)

        y = self._measurements.reshape(-1)
        x_est_flat, Sigma_post = self._gpr(
            K_inv, self.C_big, self.R_inv, x_bar, y
        )

        x_est = x_est_flat.reshape(-1, 4)
        x_last = x_est[-1]
        K_last = Sigma_post[-4:, -4:]

        x_next = F @ x_last
        K_next = F @ K_last @ F.T + Q

        return x_next, K_next, x_last, K_last

    # ------------------------------------------------------------------
    # CV dynamics
    # ------------------------------------------------------------------

    @staticmethod
    def _F_cv(dt: float) -> jnp.ndarray:
        """State transition matrix for constant-velocity model."""
        return jnp.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=jnp.float32,
        )

    @staticmethod
    def _Q_cv_single(dt: float, q_c: float) -> jnp.ndarray:
        """Per-axis process noise block (2x2) for continuous white-noise jerk."""
        return (
            jnp.array(
                [[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]],
                dtype=jnp.float32,
            )
            * q_c
        )

    def _Q_cv(self, dt: float) -> jnp.ndarray:
        """Full 4x4 process noise, block-diagonal for x and y axes."""
        Q1 = self._Q_cv_single(dt, self.q_c)
        Z = jnp.zeros((2, 2), dtype=jnp.float32)
        return jnp.block([[Q1, Z], [Z, Q1]]) + 1e-9 * jnp.eye(
            4, dtype=jnp.float32
        )

    # ------------------------------------------------------------------
    # Prior
    # ------------------------------------------------------------------

    def _init_state_xyv(self, z: jnp.ndarray) -> jnp.ndarray:
        """Initial state x0 for prior rollout from position-only measurements."""
        if z.shape[0] < 1:
            raise ValueError("TGPR_CV: empty measurement window")
        dt = float(max(self.dt, 1e-6))

        if z.shape[0] < 2:
            return jnp.array(
                [z[0, 0], z[0, 1], 0.0, 0.0], dtype=jnp.float32
            )
        vx = (z[1, 0] - z[0, 0]) / dt
        vy = (z[1, 1] - z[0, 1]) / dt
        return jnp.array([z[0, 0], z[0, 1], vx, vy], dtype=jnp.float32)

    def _prior_rollout_cv(
        self, x0: jnp.ndarray, dt: float
    ) -> jnp.ndarray:
        """Roll out the CV prior from x0 for N steps."""
        N = self._measurements.shape[0]
        x_bar = jnp.zeros((N, 4), dtype=jnp.float32)
        x_bar = x_bar.at[0, :].set(x0)
        F = self._F_cv(dt)

        def body(i, xb):
            return xb.at[i, :].set(F @ xb[i - 1, :])

        return jax.lax.fori_loop(1, N, body, x_bar)

    # ------------------------------------------------------------------
    # GP regression (JIT-compiled)
    # ------------------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _gpr(self, K_inv, C_big, R_inv, x_bar, measurements):
        x_bar = x_bar.ravel()
        measurements = measurements.ravel()
        H = K_inv + C_big.T @ R_inv @ C_big
        b = K_inv @ x_bar + C_big.T @ R_inv @ measurements
        x_est = solve(H, b)
        Sigma_post = jnp.linalg.inv(H)
        return x_est, Sigma_post

    @partial(jax.jit, static_argnums=(0,))
    def _A_lift(self, Phi: jnp.ndarray) -> jnp.ndarray:
        """Build the lifted state-transition matrix from per-step Phi blocks."""
        M, N, _ = Phi.shape
        I = jnp.eye(N, dtype=Phi.dtype)
        A = jnp.zeros((M + 1, M + 1, N, N), dtype=Phi.dtype)
        A = A.at[jnp.arange(M + 1), jnp.arange(M + 1)].set(I)

        def outer(i, A_blocks):
            Phi_im1 = Phi[i - 1]

            def inner(j, Ab):
                return Ab.at[i, j].set(Phi_im1 @ Ab[i - 1, j])

            return jax.lax.fori_loop(0, i, inner, A_blocks)

        A = jax.lax.fori_loop(1, M + 1, outer, A)
        return A.transpose(0, 2, 1, 3).reshape((M + 1) * N, (M + 1) * N)
