# Adapted from https://github.com/Yu-Wang-0923/idopNetworkAPP@fc5dc2f5a903c7fdbdadeef903771f9852f05b8c
# packages/idopnetwork/src/idopnetwork/clustering/funclu.py:91-137 (MIT License)
# and .../clustering/_v4.py:366-459 (MIT License).
# Added well before the upstream package supports it: the SAD(1) covariance
# parameters can be shared across conditions (S1-S4), so BIC can compare
# covariance structures instead of assuming S4.
"""FunClu engine with selectable SAD(1) covariance structures.

S4 (condition-specific phi and gamma) is delegated to the pinned upstream
kernel, so existing results are unchanged.  S1-S3 constrain the covariance
sharing (shared phi/gamma) in the M-step while reusing the upstream E-step,
initialization and mean-parameter updates.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from idopnetwork.clustering._v4 import FunClu as FunCluV4
from idopnetwork.clustering.funclu import FunClu as UpstreamFunClu
from idopnetwork.curve_fitting import power_fitting

COVARIANCE_MODELS = ("S1", "S2", "S3", "S4")

COVARIANCE_DESCRIPTIONS = {
    "S1": "shared phi and shared gamma across conditions",
    "S2": "shared phi, condition-specific gamma",
    "S3": "condition-specific phi, shared gamma",
    "S4": "condition-specific phi and gamma",
}


class StructuredFunCluKernel(FunCluV4):
    """Upstream SAD(1)-FunClu kernel with an S1-S4 covariance constraint."""

    def __init__(self, *args, covariance_model: str = "S4", **kwargs):
        super().__init__(*args, **kwargs)
        covariance_model = str(covariance_model).upper()
        if covariance_model not in COVARIANCE_MODELS:
            raise ValueError(f"Unknown covariance model: {covariance_model}")
        self.covariance_model = covariance_model

    def covariance_parameter_count(self) -> int:
        conditions = int(self.n_conditions_)
        return {
            "S1": 2,
            "S2": conditions + 1,
            "S3": conditions + 1,
            "S4": 2 * conditions,
        }[self.covariance_model]

    def structured_parameter_count(self) -> int:
        return (
            int(self.n_components) - 1
            + 2 * int(self.n_components) * int(self.n_conditions_)
            + self.covariance_parameter_count()
        )

    def bic(self) -> float:
        return -2.0 * float(self.log_likelihood_) + self.structured_parameter_count() * math.log(
            max(int(self.n_features_), 1)
        )

    def _m_step(self, responsibilities, params_mu, params_cov):
        if self.covariance_model == "S4":
            return super()._m_step(responsibilities, params_mu, params_cov)

        cluster_sizes = responsibilities.sum(dim=0)
        active = cluster_sizes > 1e-8
        weights = (cluster_sizes / self.n_features_).clamp_min(1e-12)
        weights /= weights.sum()
        new_mu = params_mu.clone()
        responsibility_transpose = responsibilities.T.contiguous()
        weighted_sums = responsibility_transpose @ self._combined_curves
        weighted_statistics = responsibility_transpose @ self._combined_statistics
        phi = params_cov[:, 0].clone()
        gamma = params_cov[:, 1].clamp_min(1e-8).clone()
        n_times = [float(data["n_time"]) for data in self._prepared]

        for _ in range(self.inner_max_iter):
            previous_mu = new_mu.clone()
            previous_phi, previous_gamma = phi.clone(), gamma.clone()
            residuals: list[tuple[Any, Any, Any]] = []
            for ell, data in enumerate(self._prepared):
                weighted_sum = weighted_sums[:, self._curve_slices[ell]]
                offset = 3 * ell
                weighted_y2 = weighted_statistics[:, offset]
                weighted_y2_previous = weighted_statistics[:, offset + 1]
                weighted_cross = weighted_statistics[:, offset + 2]
                mean_curves = weighted_sum / cluster_sizes[:, None].clamp_min(1e-12)
                mean_curves = torch.where(
                    active[:, None], mean_curves, torch.zeros_like(mean_curves)
                )
                mean_parameters = new_mu[:, ell].clone()
                if bool(active.any().item()):
                    mean_parameters[active] = self._fit_power_sad(
                        mean_curves[active],
                        data["z"],
                        phi[ell],
                        mean_parameters[active],
                    )
                fitted = self._evaluate_power(data["z"], mean_parameters)
                residual_ss = (
                    weighted_y2 - 2.0 * torch.sum(weighted_sum * fitted, dim=1)
                    + cluster_sizes * torch.sum(fitted.square(), dim=1)
                )
                residual_ss_previous = (
                    weighted_y2_previous
                    - 2.0 * torch.sum(weighted_sum[:, :-1] * fitted[:, :-1], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, :-1].square(), dim=1)
                )
                residual_cross = (
                    weighted_cross
                    - torch.sum(weighted_sum[:, 1:] * fitted[:, :-1], dim=1)
                    - torch.sum(weighted_sum[:, :-1] * fitted[:, 1:], dim=1)
                    + cluster_sizes * torch.sum(fitted[:, 1:] * fitted[:, :-1], dim=1)
                )
                new_mu[:, ell] = mean_parameters
                residuals.append((residual_ss, residual_ss_previous, residual_cross))

            if self.covariance_model in {"S1", "S2"}:
                numerator = sum(torch.sum(part[2]) for part in residuals)
                denominator = sum(torch.sum(part[1]) for part in residuals)
                shared = torch.where(
                    denominator > 1e-12,
                    torch.clamp(
                        numerator
                        / ((1.0 + self.phi_ridge) * denominator).clamp_min(1e-12),
                        -self.phi_bound,
                        self.phi_bound,
                    ),
                    torch.zeros_like(numerator),
                )
                phi = shared.expand_as(phi).clone()
            else:
                for ell, (_, residual_ss_previous, residual_cross) in enumerate(
                    residuals
                ):
                    denominator = torch.sum(residual_ss_previous)
                    phi[ell] = torch.where(
                        denominator > 1e-12,
                        torch.clamp(
                            torch.sum(residual_cross)
                            / ((1.0 + self.phi_ridge) * denominator).clamp_min(1e-12),
                            -self.phi_bound,
                            self.phi_bound,
                        ),
                        torch.zeros_like(phi[ell]),
                    )

            innovation = [
                residual_ss
                - 2.0 * phi[ell] * residual_cross
                + phi[ell].square() * residual_ss_previous
                + self.tightness_weight * residual_ss
                for ell, (residual_ss, residual_ss_previous, residual_cross) in enumerate(
                    residuals
                )
            ]
            if self.covariance_model in {"S1", "S3"}:
                numerator = sum(torch.sum(value) for value in innovation)
                denominator = (
                    (1.0 + self.tightness_weight)
                    * self.n_features_
                    * sum(n_times)
                )
                shared_gamma = torch.sqrt(
                    torch.clamp(numerator / denominator, min=1e-12)
                )
                gamma = shared_gamma.expand_as(gamma).clone()
            else:
                for ell, value in enumerate(innovation):
                    denominator = (
                        (1.0 + self.tightness_weight)
                        * self.n_features_
                        * n_times[ell]
                    )
                    gamma[ell] = torch.sqrt(
                        torch.clamp(torch.sum(value) / denominator, min=1e-12)
                    )

            parameter_change = torch.max(torch.abs(new_mu - previous_mu))
            covariance_change = torch.maximum(
                torch.max(torch.abs(phi - previous_phi)),
                torch.max(torch.abs(gamma - previous_gamma)),
            )
            scale = 1.0 + torch.maximum(
                torch.max(torch.abs(previous_mu)),
                torch.maximum(
                    torch.max(torch.abs(previous_phi)),
                    torch.max(torch.abs(previous_gamma)),
                ),
            )
            if float(torch.maximum(parameter_change, covariance_change).item()) <= self.inner_tol * float(scale.item()):
                break

        return weights, new_mu, torch.stack((phi, gamma), dim=1), cluster_sizes


class StructuredFunClu(UpstreamFunClu):
    """Public wrapper matching the upstream FunClu interface."""

    def __init__(self, *args, covariance_model: str = "S4", **options):
        super().__init__(*args, **options)
        covariance_model = str(covariance_model).upper()
        if covariance_model not in COVARIANCE_MODELS:
            raise ValueError(f"Unknown covariance model: {covariance_model}")
        self.covariance_model = covariance_model

    def fit(self, data, *, parameter_data=None, verbose=False):
        self._prepare_data(data)
        if parameter_data is not None and len(parameter_data) != len(data):
            raise ValueError("参数表数量必须与 condition 数量一致")
        samples: dict[str, Any] = {}
        parameters: dict[str, Any] = {}
        for index, frame in enumerate(data):
            time = np.asarray(frame.index, dtype=float)
            if (
                len(time) < 2
                or not np.isfinite(time).all()
                or (time <= 0).any()
                or np.unique(time).size < 2
            ):
                raise ValueError("聚类要求至少两个不同的有限正数时间点，请重新运行拟合器")
            if frame.columns.has_duplicates:
                raise ValueError("特征名称不能重复")
            samples[str(index)] = frame
            parameters[str(index)] = (
                power_fitting(frame)[0]
                if parameter_data is None
                else parameter_data[index]
            )
        options = dict(self.options)
        options.setdefault("compute_device", str(self.device))
        options.setdefault(
            "use_mini_batch",
            True if self.use_minibatch_kmeans is None else self.use_minibatch_kmeans,
        )
        options.setdefault("progress", verbose)
        kernel = StructuredFunCluKernel(
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            covariance_model=self.covariance_model,
            **options,
        )
        kernel._prepare_inputs(samples, parameters)
        if not 1 <= self.n_components <= kernel.n_features_:
            raise ValueError(
                f"K 必须介于 1 和有效共同特征数 {kernel.n_features_} 之间"
            )
        kernel.fit(samples, parameters)
        self.kernel_ = kernel
        self.common_cols = list(kernel.feature_names_)
        self.n_features = kernel.n_features_
        self.excluded_features_ = kernel.excluded_features_
        mu = kernel.params_mu_.copy()
        for index, time in enumerate(self.times_list):
            log_time = np.log(time)
            beta = mu[:, index, 1] / max(log_time.std(), 1e-8)
            mu[:, index, 0] *= np.exp(-beta * log_time.mean())
            mu[:, index, 1] = beta
        self.params_mu = torch.as_tensor(mu, dtype=self.dtype, device=self.device)
        self.params_cov = torch.as_tensor(
            kernel.params_cov_, dtype=self.dtype, device=self.device
        )
        self.weights = torch.as_tensor(
            kernel.weights_, dtype=self.dtype, device=self.device
        )
        self.labels = torch.as_tensor(
            kernel.labels_, dtype=torch.long, device=self.device
        )
        self.log_likelihood = kernel.log_likelihood_
        self.neg_log_likelihood = -self.log_likelihood
        self.n_params = kernel.structured_parameter_count()
        self.bic = float(kernel.bic())
        self.converged = kernel.converged_
        self.n_iter_run = kernel.n_iter_
        self.loglik_history = kernel.log_likelihood_history_.tolist()
        return self

    def covariance_spec(self) -> dict[str, Any]:
        return {
            "covariance_model": self.covariance_model,
            "description": COVARIANCE_DESCRIPTIONS[self.covariance_model],
            "n_covariance_parameters": int(self.kernel_.covariance_parameter_count()),
            "n_parameters": int(self.n_params),
        }
