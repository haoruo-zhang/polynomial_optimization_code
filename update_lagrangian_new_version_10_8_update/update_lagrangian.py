from functions_Chebyshev_basis import g_D_symbolic_coefficients_dict_p2
from functions_Chebyshev_basis import generate_lag
from functions_Chebyshev_basis import restore_matrices
from functions_Chebyshev_basis import Augmented_Lagrangian
from functions_Chebyshev_basis import update_everything
from functions_Chebyshev_basis import generate_M_d
from functions_Chebyshev_basis import generate_x_input_p2
# from functions_Chebyshev_basis import compute_function_p2
from functions_Chebyshev_basis import term_3
# from functions_Chebyshev_basis import update_Lagrangian_coefficients
import numpy as np
from functools import partial
import jax
from scipy.optimize import minimize
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # initialize the parameter
    
    d = 4
    L = 6
    D_list = list(range(2, 15))

    relative_error_value = []
    for i in range(len(D_list)):
        rho = 100
        D = D_list[i]
        orders_list, coefficients_list, polynomial, x = g_D_symbolic_coefficients_dict_p2(D)
        x_input= generate_x_input_p2(D,d,L)
        Lagrangian_coefficient = generate_lag(D,d,L)

        iteration = 0
        print("Now we begin with D = {}".format(D))

        x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
        x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)
        v_k = term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list)

        while True:
            iteration += 1
            aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L, orders_list=orders_list,
                                        coefficients_list=coefficients_list,
                                        Lagrangian_coefficient=Lagrangian_coefficient, rho=rho)
            aug_lagrangian_partial_gradient = jax.grad(aug_lagrangian_partial)
            print("-"*40)
            print("current rho = {}".format(aug_lagrangian_partial.keywords['rho']))
            result = minimize(aug_lagrangian_partial, x0=x_input,
                            method='L-BFGS-B',
                            jac=aug_lagrangian_partial_gradient,
                            options={
                                'gtol': 1e-3,             # Stopping criterion (relative gradient)
                                'ftol': 1e-5,             # Stopping criterion (absolute value)
                                'maxcor': 40,
                            })

            
            print("This is {} iteration of LBFGS".format(iteration))
            print("Minimum value of the Augmented Lagrangian function:", result.fun)
            print("Was the optimization successful?", result.success)
            print("Number of iterations:", result.nit)
            print(result.message)
            
            x_input = result.x
            Lagrangian_coefficient,rho,v_k = update_everything(x_input,rho,Lagrangian_coefficient,v_k,d,D,L)
            # Lagrangian_coefficient = update_Lagrangian_coefficients(d,D,L,x_input,Lagrangian_coefficient,rho)
            print("rho after update = {}".format(rho))


            x_mu_D_L_list,_= restore_matrices(x_input,d,D,L)
            x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)
            l_product_list = []
            for l in range(L):
                moment_product = 1
                for i in range(D):
                    moment_product *= x_M_D_L_list[i][l][0,0]
                l_product_list.append(moment_product)

            max_index = l_product_list.index(max(l_product_list))
            print(l_product_list[max_index])
            x_final= np.array([x_mu_D_L_list[i][max_index][1]/l_product_list[max_index] for i in range(D)])
            print(x_final)
            # value_final = compute_function_p2(x_final)
            value_final = polynomial(*x_final)
            print("current relative error regarding polynomial value is {}".format((value_final+ 1.3911)/ 1.3911))

            if (value_final+ 1.3911)/ 1.3911 <1e-2:
                break
            
            if iteration>100:
                break
            if rho>1e8:
                break

        print("when D = {}".format(D))
        print("relative error regarding polynomial value is {}".format((value_final+ 1.3911)/ 1.3911))
        print("-"*40)
        relative_error_value.append((value_final+1.3911)/1.3911)


    print(relative_error_value)
    plt.figure(figsize=(8, 4))
    plt.plot(D_list, relative_error_value, label="Relative Error")
    plt.yscale('log')  # Setting the y-axis to a logarithmic scale
    plt.axhline(y=1e-2, color='black', linestyle='--')  # Dashed line at y = 1e-2

    # Add labels and title
    plt.xlabel("Dimension (D)")
    plt.ylabel("Relative error")
    plt.ylim(1e-10, 1)  # Limit y-axis to show the range from 1e-6 to 1e-2

    # Show the plot
    plt.show()








