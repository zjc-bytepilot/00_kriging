"""Tests for fixed kriging system assembly and linear solving."""

from __future__ import annotations

import unittest

import numpy as np

from kriging.systems import ATPRKSystemBuilder, DSCKSystemBuilder, KrigingSolver


class KrigingSystemTest(unittest.TestCase):
    """Check matrix layout independently from image-dependent covariance kernels."""

    def test_solver_matches_numpy_solve(self) -> None:
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        rhs = np.array([[1.0], [2.0]])

        actual = KrigingSolver.solve(matrix, rhs)

        np.testing.assert_allclose(actual, np.linalg.solve(matrix, rhs), rtol=0, atol=0)

    def test_atprk_builder_adds_one_unbiasedness_constraint(self) -> None:
        system = ATPRKSystemBuilder().build(np.eye(2))

        np.testing.assert_array_equal(system.matrix[-1], np.array([1.0, 1.0, 0.0]))

    def test_dsck_builder_adds_two_unbiasedness_constraints(self) -> None:
        system = DSCKSystemBuilder().build(np.eye(2), np.zeros((2, 1)), np.eye(1))

        self.assertEqual(system.matrix.shape, (5, 5))
        np.testing.assert_array_equal(system.matrix[-2], np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
        np.testing.assert_array_equal(system.matrix[-1], np.array([0.0, 0.0, 1.0, 0.0, 0.0]))
