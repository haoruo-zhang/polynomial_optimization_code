import numpy as np
import unittest
from utils import *

# Define values used for testing 
test_values = []

# First test values
D = 1
L = 2
d = 6

# Construct moment matrices for uniform distribution over [-1,1]
M_1 = np.array([[1 / (m+n+1) if (m+n) % 2 == 0 else 0 for n in range(d+1)] for m in range(d+1)])
M_2 = np.ndarray.copy(M_1)
M = np.stack((M_1, M_2))

# Construct R for factorization M_d = R @ R.T
U, Sigma, Vh = np.linalg.svd(M_1)
eigenval, eigenvec = np.linalg.eigh(M_1)
R = U @ np.diag(np.sqrt(Sigma)) @ U.T
RRt = R @ R.T

test_values.append({
    'd' : 6,
    'D' : 1,
    'L' : 2,
    'moments' : moment_variables(M, R, RRt),
    'lambda'  : None
    })

class TestPhi(unittest.TestCase):
    def test_a(self):
        self.assertEqual(abs(10), 10)

if __name__ == '__main__':
    unittest.main()
