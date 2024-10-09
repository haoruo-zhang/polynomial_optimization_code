import numpy as np
import sympy as sp
import jax.numpy as jaxnp
import jax.numpy as jnp

#function generate polynomials
def g_D_symbolic_coefficients(D):
    """
    D is the number of variables in the polynomial
    Here we use Chebyshev basis instead of the original basis
    I simply define T_i(x)=x_i in order to extract the coefficient of each term
    """
    # Define variables x_1, x_2, ..., x_D
    x = sp.symbols(f'x1:{D+1}')
    
    # Implement the polynomial
    polynomial = (1 / D) * sum(x_i**2 for x_i in x) - sp.prod(x_i**8 for x_i in x)
    # polynomial = (1 / D) * sum(x_i**4 for x_i in x) - ((1 / D) * sum(x_i for x_i in x))**3
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


# This funciton is for restoring the matrix: x_0_M_D_L+x_1_M_D_L+x_0_R_L+x_1_R_L+x_0_M_D_1_L+x_1_M_D_1_L+x_0_S_L+x_1_S_L from the flattened x
def restore_matrices(s,d,D,L):
    """
    s is the flattend x
    d is the highest order of polynomial
    D is the number of variables in polynomial
    L is the number of measures
    """

    # Define matrix dimensions
    md_shape = (2*d+3,)
    Rd_shape = (d+1,d+1)

    # Initialize empty lists to store the restored matrices
    x_mu_D_L_list = [[] for _ in range(D)]
    x_R_L_list = [[] for _ in range(D)]
    
    # Set the initial index
    start_index = 0

    for i in range(D):
        for _ in range(L):
            # Restore list of moments of measure
            x_mu_D_L_list[i].append(s[start_index:start_index + 2*d+3].reshape(md_shape))
            start_index += 2*d+3
    

    for i in range(D):
        for _ in range(L):
            # Restore R_i(d) matrices
            x_R_L_list[i].append(s[start_index:start_index + (d+1)**2].reshape(Rd_shape))
            start_index += (d+1)**2
    
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
    return x_M_D_L_list

#Def the function of B.3
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

    #Second term
    sum_result += term_2(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,Lagrangian_coefficient)
    
    #Third term
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
        sum += Lagrangian_coefficient[1][0][l]*max(-x_M_D_L_list[0][l][0,0],0)
    
    # mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += Lagrangian_coefficient[1][i][l]*(x_M_D_L_list[i+1][l][0,0]-1)
    
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
            Lagrangian_coefficient[1][i][l] += rho*( x_M_D_L_list[i+1][l][0,0]-1)

    #8 B.2.1.
    for i in range(D):
        for l in range(L):
            Lagrangian_coefficient[2][i][l] += rho*(np.maximum(0,-x_mu_D_L_list[i][l]-1)+np.maximum(0,x_mu_D_L_list[i][l]-1))

    return Lagrangian_coefficient

def chebyshev_polynomial(n,x):
    return 1/2*sp.chebyshevt(n, x)

def definite_integrate_chebyshev(n, x, a, b):
    T_n = chebyshev_polynomial(n,x)
    definite_integral = sp.integrate(T_n, (x, a, b))
    return definite_integral

def generate_x_input(D,d,L):
    x_mu_D_L_list = [[] for _ in range(D)]
    x_R_L_list = [[] for _ in range(D)]

    # 9/28 update, we first generate a list of different moments of measure, then in the augumented lagrangian, we can generate the M_d matrix from our list of moments of measure
    # This can fix the problem that when we try to differentiate the ϕn(µ), it actully differentiate the list of moments of measure not the M_d matrix 

    # Define the mu_list
    list_size = 2*d+3
    x = sp.Symbol('x')
    my_list = np.array([definite_integrate_chebyshev(i,x,-1, 1) for i in range(list_size)])
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
            Lagrangian_coefficient[2][i].append(np.zeros(2*d+3))
        
    return Lagrangian_coefficient

def compute_function(x):
    D = len(x)
    sum_term = (1 / D) * np.sum(2 * x**2 - 1)
    product_term = np.prod(np.cos(8 * np.arccos(x)))
    result = sum_term - product_term
    return result

def update_everything(x_input,rho,Lagrangian_coefficient,v,d,D,L):
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