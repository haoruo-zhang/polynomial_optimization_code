import cProfile
from utils import *

def experiment_1():
    # polynomial is (4x_1 - 2) ** 2 + (4x_2 - 2) ** 2
    coefficients = (16, 16, -8, -8, 8)
    powers = ((2, 0),
              (0, 2),
              (1, 0),
              (0, 1),
              (0, 0))
    poly = PolySupport(coefficients, powers)
    solver(poly)

def main():
    D = 4
    poly = ExampleG(D)
    solver(poly, gamma=1_000, seed=None)

if __name__ == '__main__':
    main()
    #experiment_1()
