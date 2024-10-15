import jax.numpy as jnp
import unittest
from utils import *

class TestPhi(unittest.TestCase):
    def test_0(self):
        L = 2
        D = 1
        d = 6

        mu_vector = jnp.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = jnp.array([[jnp.copy(mu_vector) for d in range(D)] for l in range(L)])

        for i in range(d+1):
            n = (i,)
            truth = 2 / (i+1) if i % 2 == 0 else 0
            self.assertEqual(phi(n, mu, D, L), truth)

class TestPolynomial(unittest.TestCase):
    def setUp(self):
        self.ground_truth = []
        true_coefficients = [4.00000000000000,
                             1/8,
                             3/8,
                             -4.00000000000000,
                             3/8,
                             4.00000000000000,
                             1/8,
                             -4,
                             1.00000000000000]
        true_powers = [(4, 0),
                       (3, 0),
                       (2, 1),
                       (2, 0),
                       (1, 2),
                       (0, 4),
                       (0, 3),
                       (0, 2),
                       (0, 0)]
        self.ground_truth.append(PolySupport(true_coefficients, true_powers))

        true_coefficients = [8 / 3,
                             1 / 27,
                             1 / 9,
                             1 / 9,
                             -8 / 3,
                             1 / 9,
                             2 / 9,
                             1 / 9,
                             8 / 3,
                             1 / 27,
                             1 / 9,
                             -8 / 3,
                             1 / 9,
                             8 / 3,
                             1 / 27,
                             -8 / 3,
                             1.0]

        true_powers = [(4, 0, 0),
                       (3, 0, 0),
                       (2, 1, 0),
                       (2, 0, 1),
                       (2, 0, 0),
                       (1, 2, 0),
                       (1, 1, 1),
                       (1, 0, 2),
                       (0, 4, 0),
                       (0, 3, 0),
                       (0, 2, 1),
                       (0, 2, 0),
                       (0, 1, 2),
                       (0, 0, 4),
                       (0, 0, 3),
                       (0, 0, 2),
                       (0, 0, 0)]

        self.ground_truth.append(PolySupport(true_coefficients, true_powers))

    def test_0(self):
        p = ExampleG(2)
        self.assertEqual(p.coefficients, self.ground_truth[0].coefficients)
        self.assertEqual(p.powers, self.ground_truth[0].powers)

    def test_1(self):
        p = ExampleG(3)
        self.assertEqual(p.coefficients, self.ground_truth[1].coefficients)
        self.assertEqual(p.powers, self.ground_truth[1].powers)

class TestLagrangian(unittest.TestCase):
    def setUp(self):
        L = 2
        D = 2
        d = 4

        self.L = L
        self.D = D
        self.d = d

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = jnp.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = jnp.array([[jnp.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = jnp.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = jnp.zeros(M.shape)
        RRt = jnp.zeros(M.shape)

        pos_slack = jnp.ones((L, D))
        abs_slack = jnp.zeros((L, D, d+1))

        self.free_vars = FreeVariables(L, D, d, mu, R)
        # lambda
        self.lm = LagrangeMultipliers(L, D, d)


    def test_factorization_penalty(self):
        # Test that the lm.multiply() function properly evaluates the
        # lagrangian when multipliers are adjusted to align with various
        # violations of constraints (R = 0 initially)
        self.assertEqual(self.lm.multiply(self.free_vars), 0)

        self.lm.factorization = self.lm.factorization.at[0,0,3,1].set(1)
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        # Should add nothing to product, no violation here
        self.lm.factorization = self.lm.factorization.at[0,0,2,1].set(1)
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        self.lm.factorization = self.lm.factorization.at[0,0,4,2].set(-1)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 1/5 - 1/7)

        # Fix violation in M_d[0,0,3,1]
        self.free_vars.R = self.free_vars.R.at[0,0,3,0].set(1 / 5)
        self.free_vars.R = self.free_vars.R.at[0,0,1,0].set(1)
        self.free_vars.update_RRt()
        self.assertEqual(self.lm.multiply(self.free_vars), -1/7)


    def test_nonnegativity_penalty(self):
        self.lm.nonnegativity = self.lm.nonnegativity.at[1,0].set(3.7)
        self.free_vars.mu = self.free_vars.mu.at[1,0,0].set(-1.9)
        self.assertEqual(self.lm.multiply(self.free_vars), -7.03)

    def test_mu_equality_constraints(self):
        self.lm.nonnegativity = self.lm.nonnegativity.at[1,1].set(5.8)
        self.free_vars.mu = self.free_vars.mu.at[1,1,0].set(8.03)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        # should not change value
        self.lm.nonnegativity = self.lm.nonnegativity.at[0,1].set(-1_000)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        self.free_vars.mu = self.free_vars.mu.at[0,1,0].set(0.5)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 540.774)

    # TODO test moments absolute value <= 1 constraints
    # TODO redundant B.2.2 numerical stability constraint


if __name__ == '__main__':
    D = 1
    L = 2
    d = 6
    unittest.main(verbosity=2)
