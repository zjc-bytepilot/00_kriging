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
    """Build the one-constraint ATPRK support-variogram system."""

    @staticmethod
    def build(coarse_variogram: np.ndarray) -> KrigingSystem:
        count = coarse_variogram.shape[0]
        matrix = np.block(
            [
                [coarse_variogram, np.ones((count, 1))],
                [np.ones((1, count)), np.zeros((1, 1))],
            ]
        )
        return KrigingSystem(matrix=matrix)

    @staticmethod
    def rhs(fine_to_coarse_variogram: np.ndarray) -> np.ndarray:
        return np.vstack((fine_to_coarse_variogram, np.ones((1, 1))))


class DSCKSystemBuilder:
    """Build the two-constraint DSCK support-variogram system."""

    @staticmethod
    def build(
        coarse_variogram: np.ndarray,
        cross_variogram: np.ndarray,
        fine_variogram: np.ndarray,
    ) -> KrigingSystem:
        coarse_count = coarse_variogram.shape[0]
        fine_count = fine_variogram.shape[0]
        matrix = np.block(
            [
                [
                    coarse_variogram,
                    cross_variogram,
                    np.ones((coarse_count, 1)),
                    np.zeros((coarse_count, 1)),
                ],
                [
                    cross_variogram.T,
                    fine_variogram,
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
        fine_to_coarse_variogram: np.ndarray,
        fine_to_fine_variogram: np.ndarray,
    ) -> np.ndarray:
        constraints = np.array([[1.0], [0.0]])
        return np.vstack((fine_to_coarse_variogram, fine_to_fine_variogram, constraints))
