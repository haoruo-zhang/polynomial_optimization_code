import numpy as np
import sympy as sp
import torch
import jax.numpy as jaxnp
from jax import grad as jaxgrad

#function generate polynomials
def g_D_symbolic_coefficients_dict(D):
    """
    D is the number of variables in polynomial
    """
    # define variable x_1, x_2, ..., x_D
    x = sp.symbols(f'x1:{D+1}')
    
    # Implement the polynomial
    polynomial = (1 / D) * sum(8 * x_i**4 - 8 * x_i**2 + 1 for x_i in x)+(sum(x) / D) ** 3
  
    # Expand the result
    expanded_result = sp.expand(polynomial)
    
    # initialize a dict to store coefficients
    coefficients_dict = {}
    
    # Get all terms in the expanded results
    terms = expanded_result.as_ordered_terms()
    
    for term in terms:
        coeff = sp.expand(term)
        for var in x:
            coeff = coeff.coeff(var)
        variables_powers = sp.Poly(term, x).as_dict()
        
        for vars_tuple, coeff in variables_powers.items():
            coefficients_dict[vars_tuple] = coeff
    
    return expanded_result, coefficients_dict, polynomial, x


# This funciton is for restoring the matrix: x_0_M_D_L+x_1_M_D_L+x_0_R_L+x_1_R_L+x_0_M_D_1_L+x_1_M_D_1_L+x_0_S_L+x_1_S_L from the flattened x
def restore_matrices(s,d,D,L):
    """
    s is the flattend x
    d is the highest order of polynomial
    D is the number of variables in polynomial
    L is the number of measures
    """

    # Define matrix dimensions
    md_shape = (d+1, d+1)
    # Initialize empty lists to store the restored matrices
    x_M_D_L_list = [[] for _ in range(D)]
    x_R_L_list = [[] for _ in range(D)]
    
    # Set the initial index
    start_index = 0

    for i in range(D):
        for _ in range(L):
            # Restore M_d(d) matrices
            x_M_D_L_list[i].append(s[start_index:start_index + (d+1)**2].reshape(md_shape))
            start_index += (d+1)**2
    

    for i in range(D):
        for _ in range(L):
            # Restore R_i(d) matrices
            x_R_L_list[i].append(s[start_index:start_index + (d+1)**2].reshape(md_shape))
            start_index += (d+1)**2
    
    return x_M_D_L_list,x_R_L_list

def restore(s,d,D,L):
    """
    s is the flattened sequence of moment matrices and their factorizations R
    d is the degree of the polynomial
    D is the dimension of the hypercube
    L is the number of measures

    returns a reshaped numpy array, with 0th entry the moment matrices and
    1st entry the corresponding Rs
    """
    return s.reshape((2, D, L, d+1, d+1))


# Def the function of B.3
def Augmented_Lagrangian(x,d,D,L,orders_list,coefficients_list,Lagrangian_coefficient,gamma):
    """
    x is the flattend x
    D is the number of variables in polynomial
    L is the number of measures
    orders_list is the list of different terms(e.g. x1^2*x2^2) in polynomials
    coefficients_list is the list of coefficients of the above terms
    Lagrangian_coefficient is Lagrangian coefficient
    gamma is penalty parameter

    """
    #Before we start, we need to reshape the x input back to the original format, which is the matrix form

    # extract moment matrices M_d^l and their respective factorizations R where
    # M_d^l = R R^T
    matrices = restore(x,d,D,L)
    x_M_D_L_list = matrices[0]
    x_R_L_list = matrices[1]
    
    sum_result = 0

    #First term
    sum_result += objective(D,L,x_M_D_L_list,orders_list,coefficients_list)

    #Second term
    sum_result += multipliers(D,L,d,x_M_D_L_list,x_R_L_list,Lagrangian_coefficient)
    
    #Third term
    sum_result += penalty(D,L,d,x_M_D_L_list,x_R_L_list,gamma)
    return sum_result

def phi(n, M, D, L):
    """
    Calculates the sum of the product measures of a monomial given by n
    This corresponds to phi in the paper

    n is the tuple of exponents
    M is the moment matrix
    D is the dimension of the hypercube
    L is the number of product measures
    """

    # Set up array for calculations
    # Each row corresponds to a product measure
    # The entries are the moments specified by n for each component measure
    A = np.array([
        [M[0,i,l,1,n_i] for (i, n_i) in zip(range(D), n)]
        for l in range(L)])
                            
    # Multiply over the rows, then add up the resulting product measure values
    return np.sum(np.prod(A, axis=1))

#This is the sum of the polynomials
def objective(D,L,x_M_D_L_list,orders_list,coefficients_list):
    sum = 0
    for i in range(len(orders_list)):
        moments_product_sum = 0
        for l in range(L):
            moments_product = 1
            for j in range(D):
                moments_product *= x_M_D_L_list[j][l][0,orders_list[i][j]]
            moments_product_sum += moments_product
        sum +=coefficients_list[i]*moments_product_sum
    return sum

#This is the sum of the polynomials
def objective(D,L,matrices,orders_list,coefficients_list):
    """
    Returns sum of p_n \phi_n
    matrices is M_d and factorizations R
    """
    sum = 0
    for i in range(len(orders_list)):
        moments_product_sum = 0
        for l in range(L):
            moments_prodect = 1
            for j in range(D):
                moments_prodect *= x_M_D_L_list[j][l][0,orders_list[i][j]]
            moments_product_sum += moments_prodect
        sum +=coefficients_list[i]*moments_product_sum
    return sum


#Lagrangian term
def multipliers(D,L,d,x_M_D_L_list,x_R_L_list,Lagrangian_coefficient):
    
    sum = 0
    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            sum += jaxnp.sum(x_M_D_L_list[i][l]-jaxnp.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))
    

    # 5. mu_(1,0)^l>=0
    for l in range(L):
        sum += max(-x_M_D_L_list[0][l][0,0],0)
    
    # 6. mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += x_M_D_L_list[i+1][l][0,0]-1
    # #7.B.2.2
    # for l in range(L):
    #     for i in range(d+1):
    #         for j in range(d+1):
    #             product = 1
    #             for s in range(D):
    #                 product *= x_M_D_L_list[s][l][i,j]
    #             sum+= max(0,-product-1)+max(0,product-1)
    
    #8 B.2.1.
    for i in range(D):
        for l in range(L):
            sum+= jaxnp.sum(jaxnp.maximum(0,-x_M_D_L_list[i][l]-1)+jaxnp.maximum(0,x_M_D_L_list[i][l]-1))
    
    return Lagrangian_coefficient*sum

#Penanlty term
def penalty(D,L,d,x_M_D_L_list,x_R_L_list,gamma):
    sum = 0 
    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            sum += jaxnp.sum(jaxnp.square((x_M_D_L_list[i][l]-jaxnp.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))))
       
    # 5. mu_(1,0)^l>=0 here we define the penalty term as max(0, -g)**2 since we need to let it >=0 
    # P.S Here we lack a good enough method to calculate the >=0 equation
    for l in range(L):
        sum += max(0,-x_M_D_L_list[0][l][0,0])**2
    
    # 6. mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            sum += (x_M_D_L_list[i+1][l][0,0]-1)**2
    
    # #7.B.2.2
    # for l in range(L):
    #     for i in range(d+1):
    #         for j in range(d+1):
    #             product = 1
    #             for s in range(D):
    #                 product *= x_M_D_L_list[s][l][i,j]
    #             sum+= (max(0,-product-1)+max(0,product+1))**2
    
    #8 B.2.1.
    for i in range(D):
        for l in range(L):
                sum+= jaxnp.sum(jaxnp.square(jaxnp.maximum(0,-x_M_D_L_list[i][l]-1)+jaxnp.maximum(0,x_M_D_L_list[i][l]-1)))
    
    return gamma/2*sum

def update_Lagrangian_coefficients(d,D,L,x_input,Lagrangian_coefficient,gamma):
    """
    D is the number of variables in polynomial
    L is the number of measures
    x_input is the flattend x
    Lagrangian_coefficient is Lagrangian coefficient
    gamma is penalty parameter
    """
    # TODO is Lagrangian coefficient the multipliers? Why isn't it a vector?

    #Before we start, we need to reshape the x input back to the original format
    x_M_D_L_list,x_R_L_list = restore_matrices(x_input, d , D, L)
    sum = 0
    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    for i in range(D):
        for l in range(L):
            # add up all differences between M_d and its factorization R @ R.T
            # TODO do differences cancel out in this expression?
            sum += np.sum(x_M_D_L_list[i][l]-np.dot(x_R_L_list[i][l],x_R_L_list[i][l].T))
    
    # 5. mu_(1,0)^l>=0
    for l in range(L):
        # if given mu_(1,0) is less than 0, add its absolute value to sum
        sum += max(-x_M_D_L_list[0][l][0,0],0)
    
    # 6. mu_(i,0)^l - 1 = 0
    for i in range(D-1):
        for l in range(L):
            # TODO what would happen if this were less than 0? Cancel out?
            sum += x_M_D_L_list[i+1][l][0,0]-1
    
    # #7.B.2.2
    # for l in range(L):
    #     for i in range(d+1):
    #         for j in range(d+1):
    #             product = 1
    #             for s in range(D):
    #                 product *= x_M_D_L_list[s][l][i,j]
    #             sum+= max(0,-product-1)+max(0,product-1)
    
    #8 B.2.1.
    # penalize any entries with absolute value exceeding 1
    for i in range(D):
        for l in range(L):
                sum+= np.sum(np.maximum(0,-x_M_D_L_list[i][l]-1)+np.maximum(0,x_M_D_L_list[i][l]-1))
    Lagrangian_coefficient += gamma*sum
    return Lagrangian_coefficient

def auto_gradient(x,y,gamma):
    return gamma/2*jaxnp.sum(jaxnp.square((x-jaxnp.dot(y,y.T))))


def jac(x_input,d,D,L,orders_list,coefficients_list,Lagrangian_coefficient,gamma):
    """
    x is the flattend x
    D is the number of variables in polynomial
    d is the highest order in polynomial
    L is the number of measures
    orders_list is the list of different terms(e.g. x1^2*x2^2) in polynomials
    coefficients_list is the list of coefficients of the above terms
    Lagrangian_coefficient is Lagrangian coefficient
    gamma is penalty parameter

    """
    jacobian_matrix = []
    x_M_D_L_list,x_R_L_list = restore_matrices(x_input, d , D, L)

    # x_M_D_L_list
    for k in range(D):
        for l in range(L):
            for i in range(d+1):
                for j in range(d+1):
                    jacobian_matrix_sum = 0
                    measure_moment_orders = i+j
                    if i<=d and j<=d:
                        if i == 0 and j == 0 and k == 0:
                        #for 5
                            if x_M_D_L_list[k][l][i,j]<0:
                                jacobian_matrix_sum += -Lagrangian_coefficient+gamma*x_M_D_L_list[k][l][i,j]
                        #for 6                           
                        elif i == 0 and j == 0 and k!=0:
                            jacobian_matrix_sum += Lagrangian_coefficient+gamma*(x_M_D_L_list[k][l][i,j]-1)
                        #for 1
                        if measure_moment_orders<=d:
                            for o in range(len(orders_list)):
                                if orders_list[o][k]==measure_moment_orders:
                                    moments_prodect = 1
                                    for p in range(D):
                                        if p != k:
                                            moments_prodect *= x_M_D_L_list[p][l][0,orders_list[o][p]]
                                    jacobian_matrix_sum += coefficients_list[o]*moments_prodect
                        
                    #     #for 7
                    # product_1 = 1
                    # product_2 = 1
                    # for s in range(D):
                    #     if s==k:
                    #         product_1*=x_M_D_L_list[s][l][i,j]
                    #     else:
                    #         product_1*=x_M_D_L_list[s][l][i,j]
                    #         product_2*=x_M_D_L_list[s][l][i,j]
                    # if product_1<-1:
                    #     jacobian_matrix_sum += -Lagrangian_coefficient*product_2+gamma*product_2*(product_1+1)
                    # elif product_1>1:
                    #     jacobian_matrix_sum += Lagrangian_coefficient*product_2+gamma*product_2*(product_1-1)

                        #for 8
                    if x_M_D_L_list[k][l][i,j]<-1:
                        jacobian_matrix_sum += -Lagrangian_coefficient+gamma*(x_M_D_L_list[k][l][i,j]+1)
                    elif x_M_D_L_list[k][l][i,j]>1:
                        jacobian_matrix_sum += Lagrangian_coefficient+gamma*(x_M_D_L_list[k][l][i,j]-1)

                                
                    jacobian_matrix_sum += Lagrangian_coefficient
                    jacobian_matrix.append(jacobian_matrix_sum)
    
    #x_R_L_list
    for k in range(D):
        for l in range(L):
            A = torch.tensor(x_R_L_list[k][l],dtype=torch.float32,requires_grad=True)
            f = torch.matmul(A, A.T)
            f_sum = f.sum()
            f_sum.backward()
            grad = A.grad
            grad_narray = grad.detach().numpy()
            flattened_array = grad_narray.flatten()
            flattened_array *= -Lagrangian_coefficient
            jacobian_matrix = jacobian_matrix + list(flattened_array)
   
    #接下来计算的是penalty的jacob，因为过于复杂，所以引入jax计算grad
    penalty_x_M_D_L_list = []
    penalty_x_R_L_list = []
    for k in range(D):
        for l in range(L):
            M_D = x_M_D_L_list[k][l]
            R_L = x_R_L_list[k][l]
            grad_f_x = jaxgrad(auto_gradient, argnums=0)
            grad_f_y = jaxgrad(auto_gradient, argnums=1)
            M_D_flatten = np.ravel(grad_f_x(M_D, R_L,gamma)).tolist()
            R_L_flatten = np.ravel(grad_f_y(M_D, R_L,gamma)).tolist()
            penalty_x_M_D_L_list += M_D_flatten
            penalty_x_R_L_list += R_L_flatten
    
    penalty_list = penalty_x_M_D_L_list + penalty_x_R_L_list
    result = [a + b for a, b in zip(jacobian_matrix, penalty_list)]
    return(result)


