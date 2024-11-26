import cProfile
from utils import *

def main():
    D = 4
    poly = ExampleG(D)
    solver(poly, gamma=1_000, seed=None)

if __name__ == '__main__':
    #cProfile.run('main()')
    main()
