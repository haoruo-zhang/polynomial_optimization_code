import numpy as np
import sympy as sp
import jax.numpy as jaxnp
import jax.numpy as jnp
from jax import grad as jaxgrad

#function generate polynomials
def g_D_symbolic_coefficients_dict(D):
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
    
    return expanded_result, coefficients_dict, polynomial_numeric, x


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
    sum_result += term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,rho)
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
            sum += jaxnp.sum(x_M_D_L_list[i][l]-jaxnp.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))
    

    # mu_(1,0)^l>=0
    for l in range(L):
        sum += max(-x_M_D_L_list[0][l][0,0],0)
    
    # mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += x_M_D_L_list[i+1][l][0,0]-1
    
    #8 B.2.1.
    for i in range(D):
        for l in range(L):
            sum+= jaxnp.sum(jaxnp.maximum(0,-x_mu_D_L_list[i][l]-1)+jaxnp.maximum(0,x_mu_D_L_list[i][l]-1))
    
    return Lagrangian_coefficient*sum

#Penanlty term
def term_3(D,L,x_M_D_L_list,x_mu_D_L_list,x_R_L_list,rho):
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
    
    return rho/2*sum

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

    sum = 0
    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            sum += np.sum(x_M_D_L_list[i][l]-np.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))
    
    # 5. mu_(1,0)^l>=0
    for l in range(L):
        sum += max(-x_M_D_L_list[0][l][0,0],0)
    
    # 6. mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += x_M_D_L_list[i+1][l][0,0]-1

    #8 B.2.1.
    for i in range(D):
        for l in range(L):
                sum+= np.sum(np.maximum(0,-x_mu_D_L_list[i][l]-1)+np.maximum(0,x_mu_D_L_list[i][l]-1))

    Lagrangian_coefficient += rho*sum

# def auto_gradient_lag_term(x,y,Lagrangian_coefficient):
#     return Lagrangian_coefficient*jaxnp.sum(x-jaxnp.dot(y,y.T))

# def auto_gradient_penalty_term(x,y,rho):
#     return rho/2*jaxnp.sum(jaxnp.square((x-jaxnp.dot(y,y.T))))

#     return Lagrangian_coefficient
# def jac(x_input,d,D,L,orders_list,coefficients_list,Lagrangian_coefficient,rho):
#     """
#     x is the flattend x
#     D is the number of variables in polynomial
#     d is the highest order in polynomial
#     L is the number of measures
#     orders_list is the list of different terms(e.g. x1^2*x2^2) in polynomials
#     coefficients_list is the list of coefficients of the above terms
#     Lagrangian_coefficient is Lagrangian coefficient
#     rho is the penalty term 

#     """
#     jacobian_matrix = []
#     x_mu_D_L_list,x_R_L_list = restore_matrices(x_input, d , D, L)
#     x_M_D_L_list = generate_M_d(x_mu_D_L_list,d,D,L)

#     for k in range(D):
#         for l in range(L):
#             for i in range(d+1):
#                 for j in range(d+1):
#                     jacobian_matrix_sum = 0
#                     if i<=d and j<=d:
#                         if i == 0 and j == 0 and k == 0:
#                             if x_M_D_L_list[k][l][i,j]<0:
#                                 jacobian_matrix_sum += Lagrangian_coefficient+rho*x_M_D_L_list[k][l][i,j]
#                             elif x_M_D_L_list[k][l][i,j]>0:
#                                 jacobian_matrix_sum += Lagrangian_coefficient                           
#                         elif i == 0 and j == 0 and k!=0:
#                             jacobian_matrix_sum += Lagrangian_coefficient+rho*(x_M_D_L_list[k][l][i,j]-1)
#                         measure_moment_orders = i+j
#                         if measure_moment_orders<=d:
#                             for o in range(len(orders_list)):
#                                 if orders_list[o][k]==measure_moment_orders:
#                                     moments_prodect = 1
#                                     for p in range(D):
#                                         if p != k:
#                                             moments_prodect *= x_M_D_L_list[p][l][0,orders_list[o][p]]
#                                     jacobian_matrix_sum += coefficients_list[o]*moments_prodect
#                     jacobian_matrix_sum += Lagrangian_coefficient
#                     jacobian_matrix.append(jacobian_matrix_sum)
    
#     term_2_x_R_L_list = []
#     for k in range(D):
#         for l in range(L):
#             M_D = x_M_D_L_list[k][l]
#             R_L = x_R_L_list[k][l]
#             grad_f_y = jaxgrad(auto_gradient_lag_term, argnums=1)
#             R_L_flatten = np.ravel(grad_f_y(M_D, R_L, Lagrangian_coefficient)).tolist()
#             term_2_x_R_L_list += R_L_flatten
#     jacobian_matrix+=term_2_x_R_L_list
    
#     term_3_x_M_D_L_list = []
#     term_3_x_R_L_list = []
#     for k in range(D):
#         for l in range(L):
#             M_D = x_M_D_L_list[k][l]
#             R_L = x_R_L_list[k][l]
#             grad_f_x = jaxgrad(auto_gradient_penalty_term, argnums=0)
#             grad_f_y = jaxgrad(auto_gradient_penalty_term, argnums=1)
#             M_D_flatten = np.ravel(grad_f_x(M_D, R_L, rho)).tolist()
#             R_L_flatten = np.ravel(grad_f_y(M_D, R_L, rho)).tolist()
#             term_3_x_M_D_L_list += M_D_flatten
#             term_3_x_R_L_list += R_L_flatten
    
#     term_3_list = term_3_x_M_D_L_list + term_3_x_R_L_list
#     result = [a + b for a, b in zip(jacobian_matrix, term_3_list)]
#     return(result)

