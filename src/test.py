import numpy as np
import unittest
from utils import *

# Define values used for testing 
test_values = []

# First test values
D = 1
L = 2
d = 6

# Construct moment vector and matrices for uniform distribution over [-1,1]
mu_vector = [1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)]
mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                for m in range(d+1)]
                for i in range(D)]
                for l in range(L)])

# Construct R for factorization M_d = R @ R.T
U, Sigma, Vh = np.linalg.svd(M[0,0])
eigenval, eigenvec = np.linalg.eigh(M[0,0])
R = U @ np.diag(np.sqrt(Sigma)) @ U.T
RRt = R @ R.T

pos_slack = np.ones((L, D))
abs_slack = np.zeros((L, D, d+1))

var = free_variables(mu, M, R, RRt, pos_slack, abs_slack)
lm = lagrangian_vector(np.zeros((L, D, d+1, d+1)),
                       np.zeros((L, D)),
                       np.zeros((L, D, d+1)))

#polynomial = g_D_symbolic_coefficients_dict(D)

test_values.append({
    'd' : 6,
    'D' : 1,
    'L' : 2,
    'var' : var,
    'lambda'  : lm})
    #'lambda'  : lm,
    #'p' : polynomial
    #})

class TestPhi(unittest.TestCase):
    def test_0(self):
        mu = test_values[0]['var'].mu
        d = test_values[0]['d']
        D = test_values[0]['D']
        L = test_values[0]['L']

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
        self.ground_truth.append(poly_support(true_coefficients, true_powers))

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

        self.ground_truth.append(poly_support(true_coefficients, true_powers))

    def test_0(self):
        p = polynomial_g(2)
        self.assertEqual(p.coefficients, self.ground_truth[0].coefficients)
        self.assertEqual(p.powers, self.ground_truth[0].powers)

    def test_1(self):
        p = polynomial_g(3)
        self.assertEqual(p.coefficients, self.ground_truth[1].coefficients)
        self.assertEqual(p.powers, self.ground_truth[1].powers)


if __name__ == '__main__':
    D = 1
    L = 2
    d = 6
    unittest.main(verbosity=2)
