import jax
import jax.numpy as jnp
import numpy as np
import unittest
from utils import *

class TestPlots(unittest.TestCase):
    def setUp(self):
        pass

    def test_1(self):
        #dimensions = [1, 2, 3]
        dimensions = [8,]
        for D in dimensions:
            poly = PlotPoly(D)

            L = 4
            d = 6
            gamma = 8.0

            random = np.random.default_rng(seed=2958)
            # TODO mu and R
            #initial_points = random.uniform(-1, 1, size=(10, D))
            #for run in range(10):
            for run in range(1):
                initial = random.uniform(-1, 1, size=(D,))
                print('initial = {}'.format(initial))
                initial_mu = np.zeros((L, D, 2*d+1))
                for l in range(L):
                    initial_mu[l,:,:] = np.vander(initial, N=2*d+1, increasing=True)
                print('initial_mu = {}'.format(initial_mu))
                R = np.zeros((L, D, d+1, d+1))
                # TODO check this is assigning moments vertically
                for l in range(L):
                    R[l,:,:,0] = np.vander(initial, N=d+1, increasing=True)

                minimizer = solver(poly, L, gamma=gamma, max_iter=10, initial_mu=initial_mu, initial_R=R)
                print(minimizer)
                #return

    def test_2(self):
        """
        This one with PolySum
        """
        #dimensions = [1, 2, 3]
        dimensions = [3,]
        for D in dimensions:
            poly = PlotPolySum(D)

            L = 4
            d = 6
            gamma = 8.0

            random = np.random.default_rng(seed=2958)
            # TODO mu and R
            #initial_points = random.uniform(-1, 1, size=(10, D))
            #for run in range(10):
            print('D = {}'.format(D))
            for run in range(10):
                initial = random.uniform(-1, 1, size=(D,))
                print('initial = {}'.format(initial))
                initial_mu = np.zeros((L, D, 2*d+1))
                for l in range(L):
                    initial_mu[l,:,:] = np.vander(initial, N=2*d+1, increasing=True)
                print('initial_mu = {}'.format(initial_mu))
                R = np.zeros((L, D, d+1, d+1))
                # TODO check this is assigning moments vertically
                for l in range(L):
                    R[l,:,:,0] = np.vander(initial, N=d+1, increasing=True)

                minimizer, objective = solver(poly, L, gamma=gamma, max_iter=10, initial_mu=initial_mu, initial_R=R)
                print('x_min = {}'.format(minimizer))
                print('objective = {}'.format(objective))
                #return
        
