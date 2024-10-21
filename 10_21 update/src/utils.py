import numpy as np
import sympy as sp
import jax.numpy as jaxnp
import jax.numpy as jnp
from scipy.integrate import quad
from scipy.optimize import minimize
from functools import partial
import jax

def g_D_symbolic_coefficients(D):
    """
    D is the number of variables in the polynomial
    """
    # Define variables x_1, x_2, ..., x_D
    x = sp.symbols(f'x1:{D+1}')
    
    # Implement the polynomial
    polynomial = (1 / D) * sum(8 * x_i**4 - 8 * x_i**2 + 1 for x_i in x) + (sum(x) / D) ** 3
  
    # Expand the result
    expanded_result = sp.expand(polynomial)
    
    # Initialize a dict to store coefficients
    coefficients_dict = {}
    
    # Get all terms in the expanded result
    terms = expanded_result.as_ordered_terms()
    
    for term in terms:
        variables_powers = sp.Poly(term, x).as_dict()
        for vars_tuple, coeff in variables_powers.items():
            # Convert SymPy coefficients to float to be compatible with JAX
            coefficients_dict[vars_tuple] = float(coeff)
    
    # Convert the SymPy expression into a numerical function compatible with JAX
    polynomial_numeric = sp.lambdify(x, polynomial, 'numpy')
    print("Expanded Result:")
    print(expanded_result)
    print("\nCoefficients Dictionary:")
    for vars_tuple, coeff in coefficients_dict.items():
        print(f"Coefficient of {vars_tuple}: {coeff}")

    orders_list = list(coefficients_dict.keys())
    coefficients_list = list(coefficients_dict.values())
    print(orders_list)
    print(coefficients_list)
    return orders_list, coefficients_list, polynomial_numeric, x

def restore_matrices(s,d,D,L):
    """
    s is the flattend x
    d is the highest order of polynomial
    D is the number of variables in polynomial
    L is the number of measures
    """

    # Define matrix dimensions
    md_shape = (2*d+1,)
    Rd_shape = (d+1,d+1)

    # Initialize empty lists to store the restored matrices
    x_mu_D_L_list = [[] for _ in range(D)]
    x_R_L_list = [[] for _ in range(D)]
    
    # Set the initial index
    start_index = 0

    for i in range(D):
        for _ in range(L):
            # Restore list of moments of measure
            x_mu_D_L_list[i].append(s[start_index:start_index + 2*d+1].reshape(md_shape))
            start_index += 2*d+1
    

    for i in range(D):
        for _ in range(L):
            # Restore R_i(d) matrices
            x_R_L_list[i].append(s[start_index:start_index + (d+1)**2].reshape(Rd_shape))
            start_index += (d+1)**2
    x_mu_D_L_list = jnp.array(x_mu_D_L_list)
    x_R_L_list = jnp.array(x_R_L_list)
    return x_mu_D_L_list,x_R_L_list

#Def the function generting M_d matrix from mu list
def generate_M_d(x_mu_D_L_list,d,D,L):
    x_M_D_L_list = [[] for _ in range(D)]
    for l in range(L):
        for q in range(D):
            indices = jnp.arange(d + 1)
            i, j = jnp.meshgrid(indices, indices, indexing='ij')
            M_D_L_matrix = x_mu_D_L_list[q][l][i + j]
            x_M_D_L_list[q].append(M_D_L_matrix)
    return jnp.array(x_M_D_L_list)


def Augmented_Lagrangian(x_input,d,D,L,orders_list,coefficients_list,Lagrangian_coefficient,rho):
    """
    x is the flattend x
    D is the number of variables in polynomial
    L is the number of measures
    orders_list is the list of different terms(e.g. x1^2*x2^2) in polynomials
    coefficients_list is the list of coefficients of the above terms
    Lagrangian_coefficient is Lagrangian coefficient
    rho is the penalty term 

    """
    #Before we start, we need to reshape the x input back to the original format, which is the matrix form

    x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
    
    sum_result = 0

    #First term
    sum_result += term_1(D,L,x_mu_D_L_list,orders_list,coefficients_list)

    #Here we need to generate the real M_d matrix from our list of moments of measure

    x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)

    # #Second term
    sum_result += term_2(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,Lagrangian_coefficient)
    
    # #Third term
    sum_result += rho/2*term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list)
    return sum_result

def Augmented_Lagrangian_without_objective(x_input,d,D,L,Lagrangian_coefficient,rho):
    """
    x is the flattend x
    D is the number of variables in polynomial
    L is the number of measures
    orders_list is the list of different terms(e.g. x1^2*x2^2) in polynomials
    coefficients_list is the list of coefficients of the above terms
    Lagrangian_coefficient is Lagrangian coefficient
    rho is the penalty term 

    """
    #Before we start, we need to reshape the x input back to the original format, which is the matrix form

    x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
    
    sum_result = 0
    # #Here we need to generate the real M_d matrix from our list of moments of measure

    x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)

    # #Second term
    sum_result += term_2(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,Lagrangian_coefficient)
    
    # #Third term
    sum_result += rho/2*term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list)
    return sum_result

#This is the sum of the polynomials
def term_1(D,L,x_mu_D_L_list,orders_list,coefficients_list):
    sum = 0
    for i in range(len(orders_list)):
        moments_product_sum = 0
        for l in range(L):
            moments_prodect = 1
            for j in range(D):
                moments_prodect *= x_mu_D_L_list[j][l][orders_list[i][j]]
            moments_product_sum += moments_prodect
        sum +=coefficients_list[i]*moments_product_sum
    return sum

#Lagrangian term
def term_2(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,Lagrangian_coefficient):
    
    sum = 0
    # Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            sum += jaxnp.sum(Lagrangian_coefficient[0][i][l]*(x_M_D_L_list[i][l]-jaxnp.dot(x_R_L_list[i][l],x_R_L_list[i][l].T)))
    

    # mu_(1,0)^l>=0
    for l in range(L):
        sum += Lagrangian_coefficient[1][0][l]*jaxnp.maximum(-x_M_D_L_list[0][l][0,0],0)
    
    # mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += Lagrangian_coefficient[1][i+1][l]*(x_M_D_L_list[i+1][l][0,0]-1)
    
    #8 B.2.1.
    for i in range(D):
        for l in range(L):
            sum+= jaxnp.sum(Lagrangian_coefficient[2][i][l]*(jaxnp.maximum(0,-x_mu_D_L_list[i][l]-1)+jaxnp.maximum(0,x_mu_D_L_list[i][l]-1)))
    
    return sum

#Penanlty term
def term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list):
    sum = 0 
    # Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            sum += jaxnp.sum(jaxnp.square((x_M_D_L_list[i][l]-jaxnp.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))))
       
    # mu_(1,0)^l>=0 
    # here we define the penalty term as max(0, -g)**2 since we need to let it >=0 
    for l in range(L):
        sum += max(0,-x_M_D_L_list[0][l][0,0])**2
    
    # mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += (x_M_D_L_list[i+1][l][0,0]-1)**2

       
    # B.2.1.
    for i in range(D):
        for l in range(L):
                sum+= jaxnp.sum(jaxnp.square(jaxnp.maximum(0,-x_mu_D_L_list[i][l]-1)+jaxnp.maximum(0,x_mu_D_L_list[i][l]-1)))
    
    return sum

def update_Lagrangian_coefficients(d,D,L,x_input,Lagrangian_coefficient,rho):
    """
    D is the number of variables in polynomial
    L is the number of measures
    x_input is the flattend x
    Lagrangian_coefficient is Lagrangian coefficient
    rho is the penalty term 

    """

    #Before we start, we need to reshape the x input back to the original format
    x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
    x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)

    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            Lagrangian_coefficient[0][i][l] += rho*(x_M_D_L_list[i][l]-np.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))
    
    # 5. mu_(1,0)^l>=0
    for l in range(L):
        Lagrangian_coefficient[1][0][l] += rho*max(-x_M_D_L_list[0][l][0,0],0)
    
    # 6. mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            Lagrangian_coefficient[1][i+1][l] += rho*(x_M_D_L_list[i+1][l][0,0]-1)

    #8 B.2.1.
    for i in range(D):
        for l in range(L):
            Lagrangian_coefficient[2][i][l] += rho*(np.maximum(0,-x_mu_D_L_list[i][l]-1)+np.maximum(0,x_mu_D_L_list[i][l]-1))

    return Lagrangian_coefficient

def f(x,n):
    return 1/2*x**n

def generate_x_input_p2(D,d,L):
    x_mu_D_L_list = [[] for _ in range(D)]
    x_R_L_list = [[] for _ in range(D)]

    # 9/28 update, we first generate a list of different moments of measure, then in the augumented lagrangian, we can generate the M_d matrix from our list of moments of measure
    # This can fix the problem that when we try to differentiate the ϕn(µ), it actully differentiate the list of moments of measure not the M_d matrix 

    # Define the mu_list
    list_size = 2*d+1
    x = sp.Symbol('x')
    my_list = np.array([quad(f,-1,1,args=(i))[0] for i in range(list_size)])
    del x
    # # generate the list of list of both matrix
    for l in range(L):
        for i in range(D):
            x_mu_D_L_list[i].append(my_list)
            x_R_L_list[i].append(np.random.uniform(-1, 1, size=(d+1, d+1)))

    x_matrix_list_new = x_mu_D_L_list+x_R_L_list
    x_input = np.array([])
    for matrix_index in range(len(x_matrix_list_new)):
        for l in range(L):
            flattened_array = x_matrix_list_new[matrix_index][l].flatten()
            x_input = np.concatenate((x_input, flattened_array))
    x_input = jnp.array(np.array(x_input.tolist(),dtype=np.float64))
    return x_input

def generate_lag(D,d,L):
    Lagrangian_coefficient = [[[] for _ in range(D)],[[] for _ in range(D)],[[] for _ in range(D)]]

    for l in range(L):
        for i in range(D):
            Lagrangian_coefficient[0][i].append(np.zeros((d+1,d+1)))

    for l in range(L):
        for i in range(D):
            Lagrangian_coefficient[1][i].append(0)


    for l in range(L):
        for i in range(D):
            Lagrangian_coefficient[2][i].append(np.zeros(2*d+1))
        
    return Lagrangian_coefficient


def update_everything(x_input,rho,Lagrangian_coefficient,v,d,D,L):
    """
    This is the update of lagrangian coefficient and the penalty coefficient
    Using the method in <A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization> page 337-338
    """
    x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
    x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)
    v_k = term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list)

    eta = 1/4
    gamma = 10

    if v_k<eta*v:
        Lagrangian_coefficient = update_Lagrangian_coefficients(d,D,L,x_input,Lagrangian_coefficient,rho)
        v_k_1 = v_k
        print("keep rho, update Lagrangian")
    else:
        Lagrangian_coefficient = Lagrangian_coefficient
        rho = rho*gamma
        v_k_1 = v
        print("keep Lagrangian, update rho")
    return Lagrangian_coefficient,rho,v_k_1

    
def jac_term1_new(x_input, d, D, L, orders_list, coefficients_list):
    """
    Optimized version of the jac_term1_new function with index handling fixed.
    """
    jacobian_final = np.zeros(len(x_input))
    x_mu_D_L_list, _ = restore_matrices(x_input, d, D, L)
    jacobian_list = np.zeros(D * L * (2 * d + 1))
    coefficients_list = np.array(coefficients_list)
    # Precompute a mask for orders_list to avoid repeated condition checks
    orders_mask = np.array([[o[x] for x in range(D)] for o in orders_list])
    index = 0
    for x in range(D):
        for l in range(L):
            for i in range(2 * d + 1):
                if i <= d:
                    # Find terms where orders_list[o][x] == i in a vectorized way
                    relevant_terms = np.where(orders_mask[:, x] == i)[0]
                    if len(relevant_terms) == 0:
                        continue
                    
                    # Initialize moments_product_list with ones for the relevant terms
                    moments_product_list = np.ones(len(relevant_terms), dtype=float)
                    
                    # Vectorized computation of moments_product
                    for p in range(D):
                        if p != x:
                            # Ensure proper integer indexing
                            selected_orders = orders_mask[relevant_terms, p]
                            moments_product_list *= x_mu_D_L_list[p][l][selected_orders]
                    jacobian_matrix_sum = np.sum(coefficients_list[relevant_terms] * moments_product_list)
                    jacobian_list[index] = jacobian_matrix_sum
                index += 1
    
    jacobian_final[:len(jacobian_list)] = jacobian_list
    return jacobian_final



def solver(D,d,L,rho,target_value):
    """
    D is the number of dimensions
    d is the highest order in polynomial
    L is the number of measures
    rho is the value of penalty term gamma
    This function will output a global mimimum point of polynomial on [-1,1]^{D} and it's relative error subject to the real minimum value
    Lack a good stop condition and time of running is too long
    """
    orders_list, coefficients_list, polynomial, _ = g_D_symbolic_coefficients(D)
    x_input= generate_x_input_p2(D,d,L)
    Lagrangian_coefficient = generate_lag(D,d,L)

    iteration = 0
    print("Now we begin with D = {}".format(D))

    x_mu_D_L_list,x_R_L_list = restore_matrices(s=x_input,d=d,D=D,L=L)
    x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)

    # term_3 is the sum of the penalty term, but without the rho*, just the sum
    v_k = term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list)

    while True:
        iteration += 1
        # the function of whole augumented lagrangian
        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L, orders_list=orders_list,
                                    coefficients_list=coefficients_list,
                                    Lagrangian_coefficient=Lagrangian_coefficient, rho=rho)
        
        # the function of lagrangian term + penalty term
        aug_lagrangian_without_obejective_partial = partial(Augmented_Lagrangian_without_objective, d=d, D=D, L=L,
                                    Lagrangian_coefficient=Lagrangian_coefficient, rho=rho)
        
        # the gradient of the objective term
        aug_lagrangian_objective_gradient = partial(jac_term1_new,d=d,D=D,L=L,orders_list=orders_list,coefficients_list=coefficients_list)

        # the gradient of the lagrangian term + penalty term
        aug_lagrangian_without_objective_partial_gradient = jax.grad(aug_lagrangian_without_obejective_partial)

        # the gradient of the whole augumented lagrangian
        aug_lagrangian_partial_gradient = lambda x:aug_lagrangian_objective_gradient(x)+aug_lagrangian_without_objective_partial_gradient(x)

        print("-"*40)

        # the minimize function
        # Can adjust the parameter in the options
        result = minimize(aug_lagrangian_partial, x0=x_input,
                        method='L-BFGS-B',
                        jac=aug_lagrangian_partial_gradient,
                        options={
                            'gtol': 1e-5,             # Stopping criterion (relative gradient)
                            'ftol': 1e-7,             # Stopping criterion (absolute value)
                            'maxcor': 40,             # The order of the approximation Hessian
                        })

        
        print("This is {} iteration of LBFGS".format(iteration))
        print("Minimum value of the Augmented Lagrangian function:", result.fun)
        print("Was the optimization successful?", result.success)
        print("Number of iterations:", result.nit)
        print(result.message)
        
        x_input = result.x
        # We use the update rule in the Samuel Burer paper

        Lagrangian_coefficient,rho,v_k = update_everything(x_input,rho,Lagrangian_coefficient,v_k,d,D,L)

        # Calculate the x_min
        x_mu_D_L_list,_= restore_matrices(x_input,d,D,L)
        x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)
        l_product_list = []
        for l in range(L):
            moment_product = 1
            for i in range(D):
                moment_product *= x_M_D_L_list[i][l][0,0]
            l_product_list.append(moment_product)

        max_index = l_product_list.index(max(l_product_list))
        x_final= np.array([x_mu_D_L_list[i][max_index][1]/l_product_list[max_index] for i in range(D)])
        value_final = polynomial(*x_final)
        relative_error = abs((value_final-target_value)/ target_value)
        print("current x_min is {}".format(x_final))
        print("current relative error regarding polynomial value is {}".format(relative_error))

        # Some stopping conditions
        if relative_error<1e-5:
            break
        if iteration>100:
            break
        if rho>1e8:
            break

    return x_final,relative_error
