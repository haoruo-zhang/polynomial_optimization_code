import numpy as np
import jax.numpy as jnp

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

def objective_function_gradient(x_input, d, D, L, orders_list, coefficients_list):
    """
    Optimized version of the jac_term1_new function with index handling fixed.
    x_input is the flattened x
    d is the highest order in polynomial
    D is the dimension
    L is the number of measures
    orders_list is the list of order of diffenent dimension
    coefficients_list is the list of coefficients of different terms in polynomial

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