"""Fast-Funclu engine adapted to raw-scale power curves.

The estimator vendored from ``fast-funclu`` (author-provided, pinned at
583cebbeb8416678f06d979e03b7461701d1c74d) fits a linear component mean in
log-expression space.  idopNetwork Web clusters Curve Fitting exports directly
on their supplied scale, so this module overrides the mean with
``exp(alpha + beta * z)`` (a power law in tau) and keeps the SAD(1) additive
residuals on that raw scale.  The S1-S4 covariance sharing, E-step, BIC
parameter counts and EM loop remain those of Fast-Funclu.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import pandas as pd
from funclu import FunCluGMM
from sklearn.cluster import MiniBatchKMeans

_POWER_CLIP = 60.0

COVARIANCE_MODELS = ("S1", "S2", "S3", "S4")


def _apply_b(values: np.ndarray, phi: float) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    transformed = np.array(source, dtype=float, copy=True)
    if transformed.shape[-1] > 1:
        transformed[..., 1:] -= phi * source[..., :-1]
    return transformed


class PowerFunCluGMM(FunCluGMM):
    """Fast-Funclu GMM with a power mean on the supplied response scale."""

    def __init__(
        self,
        n_components: int = 1,
        *,
        tol: float = 1e-4,
        max_iter: int = 1000,
        n_init: int = 1,
        init_params: str = "log_c_beta",
        inner_max_iter: int = 1,
        beta_bounds: tuple[float, float] = (-6.0, 6.0),
        beta_xtol: float = 1e-4,
        phi_bound: float = 0.90,
        phi_ridge: float = 0.30,
        reg_covar: float = 1e-8,
        covariance_model: str = "S1",
        random_state: int = 42,
        use_minibatch_kmeans: bool = False,
    ):
        requested = str(init_params)
        base_init = "kmeans" if requested in {"log_c_beta", "c_beta"} else requested
        super().__init__(
            n_components,
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            init_params=base_init,
            random_state=random_state,
            inner_max_iter=inner_max_iter,
            beta_bounds=beta_bounds,
            beta_xtol=beta_xtol,
            phi_bound=phi_bound,
            reg_covar=reg_covar,
            covariance_model=covariance_model,
            response_transform="none",
            index_mode="sum",
        )
        if requested not in {"log_c_beta", "c_beta", "kmeans", "random"}:
            raise ValueError("init_params must be log_c_beta, kmeans or random")
        self.requested_init_params = requested
        self.init_params = requested
        self.init_params_used_ = requested
        self.init_fallback_reason_ = ""
        self.phi_ridge = float(phi_ridge)
        self.use_minibatch_kmeans = bool(use_minibatch_kmeans)

    @staticmethod
    def _linear_curve(z, alpha, beta):
        z = np.asarray(z, dtype=float)
        scalar = np.ndim(alpha) == 0 and np.ndim(beta) == 0
        alpha_array = np.atleast_1d(np.asarray(alpha, dtype=float))
        beta_array = np.atleast_1d(np.asarray(beta, dtype=float))
        exponent = np.clip(
            alpha_array[:, None] + beta_array[:, None] * z[None, :],
            -_POWER_CLIP,
            _POWER_CLIP,
        )
        curve = np.exp(exponent)
        return curve[0] if scalar else curve

    def _fit_power_profile(
        self, curve: np.ndarray, z: np.ndarray, phi: float
    ) -> np.ndarray:
        source = np.asarray(curve, dtype=float)
        transformed = _apply_b(source, phi)

        def profile(beta: float) -> tuple[float, float]:
            basis = np.exp(np.clip(beta * np.asarray(z, dtype=float), -_POWER_CLIP, _POWER_CLIP))
            transformed_basis = _apply_b(basis, phi)
            denominator = max(
                float(np.dot(transformed_basis, transformed_basis)), self.reg_covar
            )
            scale = max(
                float(np.dot(transformed_basis, transformed)) / denominator,
                self.reg_covar,
            )
            loss = float(
                np.sum((transformed - scale * transformed_basis) ** 2)
            )
            return loss, scale

        lower, upper = self.beta_bounds
        golden = 0.5 * (math.sqrt(5.0) - 1.0)
        lo, hi = float(lower), float(upper)
        first = hi - golden * (hi - lo)
        second = lo + golden * (hi - lo)
        loss_first, _ = profile(first)
        loss_second, _ = profile(second)
        while hi - lo > self.beta_xtol:
            if loss_first > loss_second:
                lo = first
                first, loss_first = second, loss_second
                second = lo + golden * (hi - lo)
                loss_second, _ = profile(second)
            else:
                hi = second
                second, loss_second = first, loss_first
                first = hi - golden * (hi - lo)
                loss_first, _ = profile(first)
        beta = first if loss_first <= loss_second else second
        _, scale = profile(beta)
        return np.array(
            [math.log(max(scale, self.reg_covar)), float(beta)], dtype=float
        )

    def _fit_linear_profile(self, curve, z, phi):
        return self._fit_power_profile(curve, z, phi)

    def _fit_linear_profiles(self, curves, z, phi):
        curves = np.asarray(curves, dtype=float)
        return np.vstack(
            [self._fit_power_profile(curve, z, phi) for curve in curves]
        )

    def _update_covariance(
        self,
        blocks,
        z_blocks,
        params_mu,
        responsibilities,
        current_phi,
        cluster_means=None,
        nk=None,
    ):
        """Fast-Funclu S1-S4 covariance update plus the phi ridge."""
        if nk is None:
            nk = responsibilities.sum(axis=0)
        if cluster_means is None:
            cluster_means = [
                responsibilities.T @ values
                / np.maximum(nk[:, None], self.reg_covar)
                for values in blocks
            ]
        data_moments = getattr(self, "_data_moments_", None)
        weighted_total = np.zeros(len(blocks), dtype=float)
        weighted_prev = np.zeros(len(blocks), dtype=float)
        weighted_cross = np.zeros(len(blocks), dtype=float)
        for ell, (values, z) in enumerate(zip(blocks, z_blocks)):
            mean_block = self._mean_block(z, params_mu, ell)
            if data_moments is None:
                values_squared = values * values
                lag_product = values[:, 1:] * values[:, :-1]
            else:
                values_squared, lag_product = data_moments[ell]
            weighted_y = nk[:, None] * cluster_means[ell]
            (
                weighted_total[ell],
                weighted_prev[ell],
                weighted_cross[ell],
            ) = self._weighted_sad_stats(
                values,
                mean_block,
                responsibilities,
                values_squared=values_squared,
                lag_product=lag_product,
                weighted_y=weighted_y,
                nk=nk,
            )
        ridge = 1.0 + float(self.phi_ridge)
        raw_phi = weighted_cross / np.maximum(
            ridge * weighted_prev, self.reg_covar
        )
        if self.covariance_model in {"S1", "S2"}:
            if self.covariance_model == "S2":
                current_phi = np.asarray(current_phi, dtype=float)
                current_ss = (
                    weighted_total
                    - 2.0 * current_phi * weighted_cross
                    + current_phi * current_phi * weighted_prev
                )
                denominators = responsibilities.shape[0] * np.asarray(
                    [values.shape[1] for values in blocks], dtype=float
                )
                precision = 1.0 / np.maximum(
                    current_ss / denominators, self.reg_covar
                )
            else:
                precision = np.ones(len(blocks), dtype=float)
            shared_phi = np.dot(precision, weighted_cross) / max(
                ridge * np.dot(precision, weighted_prev), self.reg_covar
            )
            phi = np.full(len(blocks), shared_phi, dtype=float)
        else:
            phi = raw_phi
        phi = np.clip(phi, -self.phi_bound, self.phi_bound)

        innovation_ss = (
            weighted_total
            - 2.0 * phi * weighted_cross
            + phi * phi * weighted_prev
        )
        innovation_ss = np.maximum(innovation_ss, 0.0)
        n_observations = responsibilities.shape[0]
        n_time = np.asarray([values.shape[1] for values in blocks], dtype=float)
        if self.covariance_model in {"S1", "S3"}:
            shared_gamma2 = innovation_ss.sum() / max(
                n_observations * n_time.sum(), 1.0
            )
            gamma = np.full(
                len(blocks), math.sqrt(max(shared_gamma2, self.reg_covar))
            )
        else:
            gamma2 = innovation_ss / np.maximum(n_observations * n_time, 1.0)
            gamma = np.sqrt(np.maximum(gamma2, self.reg_covar))
        return np.column_stack((phi, gamma))

    def fit(self, qd_sets: Mapping[str, pd.DataFrame], y=None):
        del y
        (
            self.condition_names_,
            self.feature_names_,
            blocks,
            z_blocks,
        ) = self._prepare_data(qd_sets)

        init_params = self.requested_init_params
        positive = all(
            np.isfinite(values).all() and bool((values > 0).all())
            for values in blocks
        )
        if init_params in {"log_c_beta", "c_beta"} and not positive:
            init_params = "kmeans"
            self.init_fallback_reason_ = (
                "log_c_beta requires strictly positive responses; using kmeans"
            )
        self.init_params = init_params
        self.init_params_used_ = init_params

        if init_params in {"log_c_beta", "c_beta"}:
            parameter_blocks = []
            for condition, values in zip(self.condition_names_, blocks):
                time = np.asarray(qd_sets[condition].index, dtype=float)
                u = self._coordinate_u(time)
                center, scale = u.mean(), u.std(ddof=0)
                design = np.column_stack((np.ones_like(u), (u - center) / scale))
                coefficients = np.linalg.lstsq(
                    design, np.log(values).T, rcond=None
                )[0].T
                beta = coefficients[:, 1] / scale
                log_c = coefficients[:, 0] - beta * center
                if not np.isfinite(log_c).all() or not np.isfinite(beta).all():
                    raise ValueError(
                        f"Non-finite log(c)/beta initialization in condition {condition!r}"
                    )
                parameter_blocks.append(np.column_stack((log_c, beta)))
            parameters = np.concatenate(parameter_blocks, axis=1)
            spread = parameters.std(axis=0, ddof=0)
            spread = np.where(spread > 0, spread, 1.0)
            self.initialization_parameters_ = parameters
            self.initialization_features_ = (parameters - parameters.mean(axis=0)) / spread

        self._data_moments_ = [
            (values * values, values[:, 1:] * values[:, :-1])
            for values in blocks
        ]
        self.n_conditions_ = len(self.condition_names_)
        self.n_features_ = len(self.feature_names_)
        self.n_features_in_ = sum(block.shape[1] for block in blocks)
        if self.n_components > self.n_features_:
            raise ValueError("n_components cannot exceed the number of feature curves")

        if self.use_minibatch_kmeans:
            if init_params in {"log_c_beta", "c_beta"}:
                cluster_data = self.initialization_features_
            elif init_params == "kmeans":
                cluster_data = np.concatenate(blocks, axis=1)
            else:
                cluster_data = None
            if cluster_data is not None:
                seed = int(
                    np.random.default_rng(self.random_state).integers(0, 2**31 - 1)
                )
                self._cached_init_labels = MiniBatchKMeans(
                    n_clusters=self.n_components,
                    n_init=3,
                    random_state=seed,
                ).fit_predict(cluster_data)

        n_runs = self.n_init
        seed_sequence = np.random.SeedSequence(self.random_state)
        rngs = [np.random.default_rng(seed) for seed in seed_sequence.spawn(n_runs)]
        best = None
        for init_index in range(n_runs):
            result = self._fit_single(
                blocks, z_blocks, rngs[init_index], init_index
            )
            if best is None or result["lower_bound"] > best["lower_bound"]:
                best = result

        self.weights_ = best["weights"]
        self.params_mu_ = best["params_mu"]
        self.params_cov_ = best["params_cov"]
        self.shared_params_cov_ = self.params_cov_.copy()
        self.responsibilities_ = best["responsibilities"]
        self.labels_ = np.argmax(self.responsibilities_, axis=1)
        self.lower_bound_ = float(best["lower_bound"])
        self.log_likelihood_ = self.lower_bound_ * self.n_features_
        self.lower_bound_history_ = best["history"]
        self.n_iter_ = int(best["n_iter"])
        self.converged_ = bool(best["converged"])
        self.cluster_sizes_ = np.bincount(
            self.labels_, minlength=self.n_components
        )
        self.effective_cluster_sizes_ = self.responsibilities_.sum(axis=0)
        self.mean_curves_ = {}
        for ell, (condition, z, frame) in enumerate(
            zip(
                self.condition_names_,
                z_blocks,
                [qd_sets[name] for name in self.condition_names_],
            )
        ):
            values = self._inverse_response(
                self._mean_block(z, self.params_mu_, ell).T
            )
            self.mean_curves_[condition] = pd.DataFrame(
                values,
                index=frame.index,
                columns=[f"cluster_{k}" for k in range(self.n_components)],
            )
        self.assignments_ = pd.DataFrame(
            {
                "feature": self.feature_names_,
                "cluster": self.labels_,
                "probability": self.responsibilities_.max(axis=1),
            }
        ).set_index("feature")
        return self


class PowerFunClu:
    """Application adapter used by the FunClu tasks."""

    def __init__(
        self,
        n_components: int = 1,
        *,
        max_iter: int = 1000,
        tol: float = 1e-4,
        inner_max_iter: int = 1,
        phi_bound: float = 0.90,
        phi_ridge: float = 0.30,
        covariance_model: str = "S1",
        init_params: str = "log_c_beta",
        use_minibatch_kmeans: bool = False,
        random_state: int = 42,
        n_init: int = 1,
    ):
        self.n_components = int(n_components)
        self.model_ = PowerFunCluGMM(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            inner_max_iter=inner_max_iter,
            phi_bound=phi_bound,
            phi_ridge=phi_ridge,
            covariance_model=covariance_model,
            init_params=init_params,
            use_minibatch_kmeans=use_minibatch_kmeans,
            random_state=random_state,
            n_init=n_init,
        )

    def fit(self, data, *, parameter_data=None, verbose=False):
        del parameter_data, verbose
        frames = {str(index): frame for index, frame in enumerate(data)}
        model = self.model_.fit(frames)
        self.common_cols = [str(column) for column in model.feature_names_]
        self.n_features = int(model.n_features_)
        self.n_conditions = int(model.n_conditions_)
        self.labels = np.asarray(model.labels_, dtype=int)
        self.weights = np.asarray(model.weights_, dtype=float)
        self.params_cov = np.broadcast_to(
            np.asarray(model.params_cov_, dtype=float)[None],
            (self.n_components, self.n_conditions, 2),
        ).copy()
        self.params_mu = self._convert_mu(model.params_mu_, data)
        self.log_likelihood = float(model.log_likelihood_)
        self.neg_log_likelihood = -self.log_likelihood
        self.bic = float(model.bic())
        self.n_params = int(model._n_parameters())
        self.n_covariance_parameters = int(
            model.covariance_spec()["n_covariance_parameters"]
        )
        self.converged = bool(model.converged_)
        self.n_iter_run = int(model.n_iter_)
        self.covariance_model = model.covariance_model
        self.init_params_used = str(model.init_params_used_)
        self.init_fallback_reason = str(model.init_fallback_reason_)
        self._condition_names = list(model.condition_names_)
        self._mean_curves = model.mean_curves_
        return self

    @staticmethod
    def _convert_mu(params_mu, data) -> np.ndarray:
        converted = np.asarray(params_mu, dtype=float).copy()
        for index, frame in enumerate(data):
            time = np.asarray(frame.index, dtype=float)
            u = np.log(time)
            spread = max(float(u.std(ddof=0)), 1e-8)
            b = converted[:, index, 1] / spread
            a = np.exp(
                np.clip(converted[:, index, 0] - b * u.mean(), -700.0, 700.0)
            )
            converted[:, index, 0] = a
            converted[:, index, 1] = b
        return converted

    def get_cluster_curves(self, condition_idx=0):
        frame = self._mean_curves[self._condition_names[condition_idx]]
        return frame.index.to_numpy(float), frame.to_numpy(float).T

    def covariance_spec(self) -> dict[str, Any]:
        spec = dict(self.model_.covariance_spec())
        spec["init_params_used"] = self.init_params_used
        return spec
