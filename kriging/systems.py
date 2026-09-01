"""Fixed kriging-system assembly shared by ATPRK and DSCK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class KrigingSystem:
    """A coefficient matrix that remains fixed across subpixel RHS values."""

    matrix: np.ndarray


class KrigingSolver:
    """Solve a kriging system without explicitly forming a matrix inverse."""

    @staticmethod
    def solve(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(matrix, rhs)


class ATPRKSystemBuilder:
    """Build the one-constraint ATPRK coefficient system."""

    @staticmethod
    def build(covariance: np.ndarray) -> KrigingSystem:
        count = covariance.shape[0]
        matrix = np.block(
            [
                [covariance, np.ones((count, 1))],
                [np.ones((1, count)), np.zeros((1, 1))],
            ]
        )
        return KrigingSystem(matrix=matrix)

    @staticmethod
    def rhs(fine_to_coarse_covariance: np.ndarray) -> np.ndarray:
        return np.vstack((fine_to_coarse_covariance, np.ones((1, 1))))


class DSCKSystemBuilder:
    """Build the two-constraint DSCK block coefficient system."""

    @staticmethod
    def build(
        coarse_covariance: np.ndarray,
        cross_covariance: np.ndarray,
        fine_covariance: np.ndarray,
    ) -> KrigingSystem:
        coarse_count = coarse_covariance.shape[0]
        fine_count = fine_covariance.shape[0]
        matrix = np.block(
            [
                [
                    coarse_covariance,
                    cross_covariance,
                    np.ones((coarse_count, 1)),
                    np.zeros((coarse_count, 1)),
                ],
                [
                    cross_covariance.T,
                    fine_covariance,
                    np.zeros((fine_count, 1)),
                    np.ones((fine_count, 1)),
                ],
                [
                    np.ones((1, coarse_count)),
                    np.zeros((1, fine_count)),
                    np.zeros((1, 2)),
                ],
                [
                    np.zeros((1, coarse_count)),
                    np.ones((1, fine_count)),
                    np.zeros((1, 2)),
                ],
            ]
        )
        return KrigingSystem(matrix=matrix)

    @staticmethod
    def rhs(
        fine_to_coarse_covariance: np.ndarray,
        fine_to_fine_covariance: np.ndarray,
    ) -> np.ndarray:
        constraints = np.array([[1.0], [0.0]])
        return np.vstack((fine_to_coarse_covariance, fine_to_fine_covariance, constraints))
