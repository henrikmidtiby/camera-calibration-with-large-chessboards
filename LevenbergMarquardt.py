from typing import Callable, Tuple

import numpy as np
from icecream import ic
from scipy.stats import chi2


class LevenbergMarquardt:
    def __init__(
        self,
        function: Callable[[np.ndarray], np.ndarray],
        initial_param: np.ndarray,
        damping: float = 10,
    ) -> None:
        self.parameters_to_optimize = initial_param == initial_param
        self.damping = damping
        self.func = function
        self.param = initial_param
        self.residual_error = np.linalg.norm(self.func(self.param))

    def jacobian(
        self, param: np.ndarray, func: Callable[[np.ndarray], np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        e = 0.1
        delta = np.zeros(param.shape)

        # Calculate the function values at the given pose
        projected_points = func(param)

        # Calculate jacobian by perturbing the pose prior
        # to calculating the function values.
        j = np.zeros((projected_points.shape[1], param.shape[0]))
        for k in range(param.shape[0]):
            delta_k = delta.copy()
            delta_k[k] = e
            param_temp = param + delta_k.transpose()
            func_value = func(param_temp)
            j[:, k] = (func_value - projected_points) / e

        # Limit the jacobian to the parameters that should be optimized.
        j = j[:, self.parameters_to_optimize]
        return (projected_points, j)

    def iterate(self) -> None:
        # Get projection errors and the associated jacobian
        self.projection_errors, j = self.jacobian(self.param, self.func)

        # Levenberg Marquard update rule
        self.coefficient_covariance_matrix = j.transpose() @ j
        t2 = np.diag(np.diag(self.coefficient_covariance_matrix)) * self.damping
        t3 = self.coefficient_covariance_matrix + t2
        param_update = (
            np.linalg.inv(t3) @ j.transpose() @ self.projection_errors.transpose()
        )

        # Unpack to full solution
        dx = np.zeros((self.param.shape[0], 1))
        dx[self.parameters_to_optimize] = param_update
        updated_x = self.param - dx.reshape((-1))
        updated_residual_error = np.linalg.norm(self.func(updated_x))

        if self.residual_error < updated_residual_error:
            # Squared error was increased, reject update and increase damping
            self.damping = self.damping * 10
        else:
            # Squared error was reduced, accept update and decrease damping
            self.param = updated_x
            self.damping = self.damping / 3
            self.residual_error = updated_residual_error

        return

    def estimate_uncertainties(self, p=0.6827):
        self.squared_residual_error = self.residual_error**2
        number_of_observations = self.projection_errors.size
        number_of_parameters = self.parameters_to_optimize.size
        # https://www.youtube.com/watch?v=3IgIToOV2Wk at 4:39
        sigma_hat_squared = self.squared_residual_error / (
            number_of_observations - number_of_parameters
        )
        # ic(sigma_hat_squared)

        # Determine how many standard deviations we should go out
        # to cover a given probability (p).
        # TODO: I am unsure if it should be split into these two cases (one_dim vs multi_dim)
        self.scale_one_dim = chi2.ppf(p, 1)
        self.scale_multi_dim = chi2.ppf(p, self.param.size)
        # ic(self.scale_one_dim)
        # ic(self.scale_multi_dim)

        # Equation 15.4.15 from Numerical Recipes in C 2002
        self.param_uncert = (
            self.scale_one_dim
            * np.sqrt(sigma_hat_squared)
            / np.sqrt(np.diag(self.coefficient_covariance_matrix))
        )
        # ic(self.param_uncert)

        # Equation on page 660 in Numerical Recipes in C 2002
        self.goodness_of_fit = 1 - chi2.cdf(
            self.residual_error**2, self.projection_errors.size
        )

        # Build matrix with uncertainties for independent parameters
        # Equation 15.4.15 from Numerical Recipes in C 2002
        delta = np.zeros(self.param.shape)
        self.independent_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            delta_k = delta.copy()
            delta_k[k] = 1
            vector = self.param_uncert * delta_k
            self.independent_uncertainties[k, :] = vector
        # ic(self.independent_uncertainties)

        # Build matrix with uncertainties for combined parameters
        # Based on equation 15.4.18 in Numericap Recipes in C 2002
        u, s, vh = np.linalg.svd(np.linalg.inv(self.coefficient_covariance_matrix))
        self.combination_uncert = self.scale_multi_dim * np.sqrt(s * sigma_hat_squared)
        self.combined_uncertainties = np.zeros((self.param.size, self.param.size))
        for k in range(self.param.size):
            vector = self.scale_multi_dim * vh[k] * np.sqrt(s[k] * sigma_hat_squared)
            self.combined_uncertainties[k, :] = vector
        # ic(self.combined_uncertainties)
        val = np.abs(self.combined_uncertainties)
        self.combined_uncertainties_vector = np.max(val, axis=0)
