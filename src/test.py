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

if __name__ == '__main__':
    D = 1
    L = 2
    d = 6
    #n = (3,)
    #print(phi(n, test_values[0]['var'].mu, D, L))
    #n = (2,)
    #print(phi(n, test_values[0]['var'].mu, D, L))
    #n = (6,)
    #print(phi(n, test_values[0]['var'].mu, D, L))
    #n = (5,)
    #print(phi(n, test_values[0]['var'].mu, D, L))
    unittest.main(verbosity=2)
