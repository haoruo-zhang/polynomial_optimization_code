import jax
import jax.numpy as jnp
import numpy as np
import unittest
from utils import *
import matplotlib.pyplot as plt
import scienceplots

class TestPlots(unittest.TestCase):
    def setUp(self):
        pass

    def test_fig(self):
        #gd_objective = np.load('gd_objective_values.npy')[:6,:]
        gd_objective = np.array([0.00537109, 0.01253255, 0.01700846, 0.02416992, 0.02596029,
       0.03222656])
        #objective = np.load('objective_values-D1-6_precisest.npy')
        objective = np.abs(np.load('objective_values.npy'))
        #objective = np.expand_dims(objective, axis=1)
        #print(objective_D4)
        #print(objective_rest)
        print(objective)
        #obj_avg = np.average(objective, axis=1)
        obj_avg = objective
        print(obj_avg)
        #gd_avg = np.average(gd_objective, axis=1)
        gd_avg = gd_objective
        print()
        print(gd_objective)
        print(gd_avg)
        D = np.array([1, 2, 3, 4, 5, 6])

        plt.style.use('science')
        plt.figure()
        purple = (120/255, 94/255, 240/255)
        yellow = (255/255, 176/255, 0/255)

        local_min = np.array([9e-3, 9e-3, 9e-3, 9e-3, 9e-3, 9e-3])
        plt.plot(D, gd_avg, label='Gradient descent', color=yellow, linestyle='--', linewidth=2.5)
        plt.plot(D, obj_avg, label='Our method', color=purple, linewidth=2.5)
        plt.plot(D, local_min, label='Local minima', color='black', linestyle='--', linewidth=2.5)

        plt.yscale('log')

        # Labels and legend
        plt.xlabel('Dimension $n$')
        plt.ylabel('Objective gap')
        #plt.legend(loc='center left', fontsize='small')
        plt.legend(loc='center left', fontsize='small', borderpad=2.0)
        #plt.grid(True, which='both', linestyle='--', linewidth=0.5)

        #plt.tight_layout()
        #plt.show()
        plt.savefig('figure.pdf')


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
        #dimensions = [1, 2, 3, 4, 5, 6, 7, 8]
        #dimensions = [5, 6, 7, 8, 9, 10]
        #dimensions = [1, 2, 3, 4]
        dimensions = [7, 8, 9]
        print(dimensions[0])
        print(dimensions[-1])
        random = np.random.default_rng(seed=2958)
        objective_results = np.zeros((len(dimensions), 1))
        for i, D in enumerate(dimensions):
            poly = PlotPolySum(D)

            #L = 4
            #L = max(4, int(D / 2) + 3)
            #L = int(D / 2 + 3)
            L = 6 # back to basic L for this one
            d = 6
            gamma = 8.0
            obj_epsilon = 1e-8 # will be scaled with dimension

            # TODO mu and R
            #for run in range(10):
            print('D = {}'.format(D))
            for run in range(1):
                #initial = random.uniform(-1, 1, size=(D,))
                initial_mu = np.zeros((L, D, 2*d+1))
                R = np.zeros((L, D, d+1, d+1))
                for l in range(L):
                    # not currently used
                    #initial = random.uniform(-1, 1, size=(D,))
                    #print('l = {}'.format(l))
                    #print('initial = {}'.format(initial))
                    uniform = np.zeros((2*d + 1,))
                    for k in range(2*d+1):
                        if k % 2 == 0:
                            uniform[k] = 1 / (2*k+1)
                    print('D = {} uniform = {}'.format(D, uniform))
                    initial_mu[l,:,:] = uniform
                    #initial_mu[l,:,:] = np.vander(initial, N=2*d+1, increasing=True)
                    #R[l,:,:,0] = np.vander(initial, N=d+1, increasing=True)
                    R[l,:,:,0] = uniform[:d+1]
                #print('initial_mu = {}'.format(initial_mu))
                # TODO check this is assigning moments vertically
                #for l in range(L):
                #    R[l,:,:,0] = np.vander(initial, N=d+1, increasing=True)

                #minimizer, objective = solver(poly, L, gamma=gamma, max_iter=10 * D, initial_mu=initial_mu, initial_R=R, verbose=False, epsilon=obj_epsilon ** D, multiplier=4)
                #minimizer, objective = solver(poly, L, gamma=gamma, max_iter=20, initial_mu=initial_mu, initial_R=R, verbose=False, epsilon=obj_epsilon ** D, multiplier=4)
                minimizer, objective = solver(poly, L, gamma=gamma, max_iter=40, initial_mu=initial_mu, initial_R=R, verbose=False, epsilon=obj_epsilon, multiplier=4)
                print('x_min = {}'.format(minimizer))
                print('x_min[0] = {}'.format(minimizer[0]))
                print('objective = {}'.format(objective))
                objective_results[i,run] = objective
        
        np.save('objective_values_D{}-{}.npy'.format(dimensions[0], dimensions[-1]), objective_results)
        print('\nObjective results in array form')
        print(objective_results)
