from src.utils import solver

if __name__ == "__main__":
    # initialize the parameter
    d = 4
    L = 6
    D = 2
    rho = 1000
    target_value = -1.3911457

    x_final,relative_error = solver(D,d,L,rho,target_value)
    print(x_final,relative_error)