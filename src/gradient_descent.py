import numpy as np
#import matplotlib.pyplot as plt

def poly(x):
    values = 10 * (x * x) * ((9 / 512.0) - (25.0 / 256) * (x * x) + (1.0 / 6) * (x * x * x * x))
    return np.sum(values)

def grad(x):
    values = 10 * x * ((18 / 512.0) - (100.0 / 256) * (x * x) + (x * x * x * x))
    return values
    

random = np.random.default_rng(seed=23491)
objective_values = np.zeros((10,10))
for D in [1, 2, 3, 4, 5, 6]:
    print('\nD = {}'.format(D))
    #stepsize = 1e-2
    #max_iter = 100_000
    stepsize = 1e-1
    max_iter = 1_000
    epsilon = 1e-10 * np.sqrt(D)
    initials = random.uniform(-1, 1, size=(D, 10))
    print('\n\n\n')
    print(epsilon)
    print(initials)
    #for n in range(10):
    for n in range(10):
        x = initials[:,n]
        for i in range(max_iter):
            #print('x = {}'.format(x))
            #print('f(x) = {}'.format(poly(x)))
            direction = -1 * grad(x)
            #print('direction = {}'.format(direction))
            #print('|direction| = {}'.format(np.linalg.norm(direction)))
            x += direction * stepsize
            if np.linalg.norm(direction) < epsilon:
                print('termination condition satisfied')
                break

        print('final x = {}\nf(x) = {}'.format(x, poly(x)))
        objective_values[D-1,n] = poly(x)


print(objective_values)
np.save('exp_gd_objective_values.npy', objective_values)
#print(poly(np.array([0.4, 0.001, -0.2]).T))
#print(poly(np.array([1, 1, 1]).T))
#print(grad(np.array([0.4, 0.001, -0.2]).T))
#print(grad(np.array([1, 1, 1]).T))
