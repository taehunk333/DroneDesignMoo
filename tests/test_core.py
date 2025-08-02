import unittest
import numpy as np

from DroneDesignMoo.problem import DroneOptimizationProblem
from DroneDesignMoo.postprocess import decode_discrete_vars

class TestDroneOptimizationProblem(unittest.TestCase):
    def setUp(self):
        self.problem = DroneOptimizationProblem()

    def test_num_variables(self):
        self.assertTrue(self.problem.n_var > 0)

    def test_evaluate_shape(self):
        # Create a random solution
        X = np.random.rand(1, self.problem.n_var)
        F, G = self.problem.evaluate(X)
        self.assertEqual(F.shape[0], 1)
        self.assertEqual(G.shape[0], 1)

    def test_constraints(self):
        # Check that constraints are evaluated
        X = np.random.rand(1, self.problem.n_var)
        _, G = self.problem.evaluate(X)
        self.assertEqual(G.shape[1], self.problem.n_constr)

class TestPostprocess(unittest.TestCase):
    def test_decode_discrete_vars(self):
        # Provide enough columns for decode_discrete_vars (at least 9)
        X = np.zeros((2, 9))  # 2 samples, 9 variables
        X_decoded = decode_discrete_vars(X)
        self.assertEqual(X_decoded.shape, X.shape)

if __name__ == '__main__':
    unittest.main()
