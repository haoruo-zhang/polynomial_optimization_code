from collections import namedtuple
from functools import partial
import jax.numpy as jnp
from jax import grad as jaxgrad
import numpy as np
import sympy as sp
import torch

# The support of the polynomial objective function
# coefficients - real-valued p_n for each monomial term
# powers - sequences n of multi-indexes for each monomial x^n
class PolySupport:
    def __init__(self, coefficients, powers):
        self.coefficients = coefficients
        self.powers = powers

# Polynomial support of the polynomial in example 3.2 of Letourneau et al. 2024
class ExampleG(PolySupport):
    """
    Generates the polynomial $g_D(x)$ from example 3.2 in Letourneau paper, in
    given dimension D
    """
    def __init__(self, D):
        # define variable x_1, x_2, ..., x_D
        x = sp.symbols(f'x1:{D+1}')
        
        # Create the polynomial
        polynomial = (1 / D) * sum(8 * x_i**4 - 8 * x_i**2 + 1 for x_i in x)+(sum(x) / D) ** 3
      
        # Expand the result
        expanded_result = sp.expand(polynomial)
        
        # Get all terms in the expanded results
        terms = expanded_result.as_ordered_terms()

        coefficients = []
        powers = []
        
        for term in terms:
            monomial = sp.Poly(term, x)

            # translate coefficient to floating point from sympy format
            coef = float(sp.polys.polytools.LC(monomial))
            coefficients.append(coef)

            n = sp.degree_list(monomial)
            powers.append(n)

        super().__init__(coefficients, powers)


# Define Lagrange multipliers structure, shaped to match up with the
# different matrices for which it is penalizing constraints.
# See (B.1) and section B.2.1 of Letourneau et al. for details
# factorization - elementwise equality constraints for M_d = R @ R.T
# nonnegativity - mu_{1, 0} >= 0, and mu_{i,0} == 1 for all i \in [2,D]
#                 and all product measures mu^l
# relaxation    - |mu_{i,n_i}^l| <= 1 for all l, i <= D. See section B.2.1
class LagrangeMultipliers:
    def __init__(self, L, D, d):
        self.L = L
        self.D = D
        self.d = d

        self.factorization = jnp.zeros((L, D, d+1, d+1))
        self.nonnegativity = jnp.zeros((L, D))
        self.relaxation = jnp.zeros((L, D, d+1))

    def multiply(self, free_vars):
        """
        Evaluate Lagrange multipliers multiplied by the constraints, based on
        the passed free variables
        """
        total = 0
        # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
        # Penalize inaccurate factorizations

        free_vars.update_RRt()
        # TODO update M_d as well?

        # add up Lagrange multipliers times each componentwise difference
        # between M_d and R @ R.T
        total += jnp.einsum('abij,abij->', free_vars.M_d - free_vars.RRt, self.factorization)
        
        # 5. mu_(1,0)^l>=0, so anything positive is clipped
        total += jnp.minimum(free_vars.mu[:,0,0], 0) @ self.nonnegativity[:,0]

        # 6. mu_(i,0)^l - 1 = 0
        total += jnp.einsum('ij,ij->',
                      free_vars.mu[:,1:,0].reshape(self.L, self.D-1) -
                      jnp.ones((self.L, self.D-1)),
                      self.nonnegativity[:,1:])

        # B.2.1. check that relevant moments have absolute value at most 1
        A = jnp.maximum(jnp.abs(free_vars.mu[:,:,:self.d+1]) -
                       jnp.ones((self.L, self.D, self.d+1)), 0)
        total += jnp.einsum('ijk,ijk->', A, self.relaxation)

        # TODO redundant B.2.2 numerical stability constraint
        return total

# Define data structure for the moment matrices M_d and their factorizations
# R such that M_d = R @ R.T. This equality does not always hold during execution
# of the algorithm, but differences will be penalized in the Lagrangian
# M_d   - a tensor containing the (d+1) x (d+1) moment matrix for each
#         of the D component measures for each of the L product measures.
#         So an L x D x (d+1) x (d+1) tensor
# R     - tensor containing factors R such that M_d = R @ R.T for each of the
#         component measures in M_d
# RRt   - tensor containing R @ R.T for each R. This is updated when we evaluate
#         the Lagrangian after updating R
# pos_slack - slack variables to enforce product measure constraints:
#              positivity constraints on mu_1,0^l and mu_i,0 = 1 constraints for i = 2, ...,
#              D
# abs_slack  - slack variable to enforce |mu_{i,n_i}^l| <= 1 from B.2.1
class FreeVariables:
    def __init__(self, L, D, d, mu=None, R=None, seed=None):
        random = np.random.default_rng(seed) # None will yield OS-selected seed

        self.mu = jnp.array(mu) if mu is not None else random.random(
                size=(L, D, 2 * d +1)) * 2 - jnp.ones((L, D, 2*d+1))
        self.M_d = jnp.array([[[[mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])

        if R is not None:
            self.R = jnp.array(R)
        else:
            random_R = random.random(size=(L, D, d+1, d+1)) * 2 - jnp.ones((L, D, d+1, d+1))
            self.R = random_R

        # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
        self.update_RRt()

        # TODO reevaluate and remove these?
        self.pos_slack = jnp.ones((L, D))
        self.abs_slack = jnp.zeros((L, D, d+1))

    def update_RRt(self):
        # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
        self.RRt = jnp.einsum('abik,abjk->abij', self.R, self.R)

    def update_M_d(self):
        self.M_d = jnp.array([[[[self.mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])

def multiply_lagrangian(l_factorization, l_nonnegativity, l_relaxation,
                        mu, M_d, R, L, D, d):
    """
    Method to multiply lagrange multiplier vector by the infeasibilities.
    Separated out from the LagrangeMultipliers class for use by jax autogradient
    """
    # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
    RRt = jnp.einsum('abik,abjk->abij', R, R)
    #RRt = jnp.inner(R, R) # = R @ R.T
    total = 0

    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    # Penalize inaccurate factorizations
    # add up Lagrange multipliers times each componentwise difference
    # between M_d and R @ R.T
    total += jnp.einsum('abij,abij->', M_d - RRt, l_factorization)
    
    # 5. mu_(1,0)^l>=0, so anything positive is clipped
    total += jnp.minimum(mu[:,0,0], 0) @ l_nonnegativity[:,0]

    # 6. mu_(i,0)^l - 1 = 0
    total += jnp.einsum('ij,ij->',
                  mu[:,1:,0].reshape(L, D-1) -
                  jnp.ones((L, D-1)),
                  l_nonnegativity[:,1:])

    # B.2.1. check that relevant moments have absolute value at most 1
    A = jnp.maximum(jnp.abs(mu[:,:,:d+1]) -
                   jnp.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, l_relaxation)

    # TODO redundant B.2.2 numerical stability constraint
    return total

def new_penalty(mu, M_d, R, gamma, L, D, d):
    """
    Calculate penalty term by adding up squared infeasibilities
    """
    # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
    RRt = jnp.einsum('abik,abjk->abij', R, R)
    #RRt = jnp.inner(R, R) # = R @ R.T
    total = 0

    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    # Penalize inaccurate factorizations
    diff = M_d - RRt
    total += jnp.einsum('abij,abij->', diff, diff)
    
    # 5. mu_(1,0)^l>=0, so anything positive is clipped
    negatives = jnp.minimum(mu[:,0,0], 0)
    total += negatives @ negatives

    # 6. mu_(i,0)^l - 1 = 0
    diff = mu[:,1:,0].reshape(L, D-1) - jnp.ones((L, D-1))
    total += jnp.einsum('ij,ij->', diff, diff)

    # B.2.1. check that relevant moments have absolute value at most 1
    A = jnp.maximum(jnp.abs(mu[:,:,:d+1]) -
                   jnp.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, A)

    # TODO redundant B.2.2 numerical stability constraint
    return (gamma / 2) * total

def grad_penalty_mu(mu, M_d, R, gamma, L, D, d):
    """
    Calculate gradient of penalty with respect to mu
    """
    # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
    RRt = np.einsum('abik,abjk->abij', R, R)
    #RRt = jnp.inner(R, R) # = R @ R.T
    result = np.zeros((L, D, 2 * d + 1))

    # factorization infeasibilities, linear in mu for each
    # individual matrix term (one mu may occupy multiple terms in M_d)
    # this is following the cross-diagonal form of M_d, that (M_d)_{a,b} =
    # mu_{a+b}
    diff = M_d - RRt
    for n_i in range(2*d + 1):
        lower = max(0, n_i - d)
        number = (d+1) - abs(d - n_i)
        upper = lower + number
        for k in range(number):
            result[:,:,n_i] += diff[:,:,lower+k,upper-1-k]

    # nonnegativity infeasibilities, >= 0 and == 1 for i = 2, ..., D
    # highlight infeasible mu_1,0 (in this case, negatives)
    infeas = np.copy(mu[:,0,0])
    infeas[infeas >= 0] = 0
    result[:,0,0] += infeas

    # gradient for mu_i,0 for i = 2, ..., D (constraint is == 1)
    # this is just the Lagrange multiplier because constraint is linear
    # function of mu
    result[:,1:,0] += mu[:,1:,0] - np.ones((L, D-1))

    # gradient for the relaxation constraint from B.2.1, restricting absolute
    # values to <= 1
    # gradient is sign(mu) * gamma * (|mu| - 1) if |mu| > 1, 0 otherwise
    # gamma is saved until the end
    A = np.maximum(np.abs(mu[:,:,:d+1]) -
                   np.ones((L, D, d+1)), 0)
    result[:,:,:d+1] += np.sign(mu[:,:,:d+1]) * A

    return gamma * result

def grad_penalty_R(mu, M_d, R, gamma, L, D, d):
    """
    Calculate gradient of penalty with respect to R
    """
    # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
    RRt = np.einsum('abik,abjk->abij', R, R)
    #RRt = jnp.inner(R, R) # = R @ R.T
    result = np.zeros((L, D, d+1, d+1))

    for i in range(d+1):
        for j in range(d+1):
            # Add up the partials w/r/t R_{i,j} from each
            # term in column i or R @ R.T
            for x in range(d+1):
                term = np.copy(mu[:,:,i+x])
                term -= np.einsum('abk,abk->ab', R[:,:,i,:], R[:,:,x,:])
                term *= -2 * R[:,:,x,j]
                result[:,:,i,j] += term


            # Add up the partials w/r/t R_{i,j} from each
            # term in row i of R @ R.T (double-counting just yields the
            # correct derivative)
            # TODO maybe the double-counting causes a problem here?
            for y in range(d+1):
                term = np.copy(mu[:,:,i+y])
                term -= np.einsum('abk,abk->ab', R[:,:,i,:], R[:,:,y,:])
                term *= -2 * R[:,:,y,j]
                result[:,:,i,j] += term


    return (gamma / 2) * result

def grad_R(l_factorization, l_nonnegativity, l_relaxation,
                        mu, R, L, D, d):
    """
    Returns gradients of the Lagrange multipliers term of the Lagrangian, with
    respect to the factorization matrix R
    """
    result = -1 * np.einsum('abik,abkj->abij', l_factorization, R)
    result += -1 * np.einsum('abki,abkj->abij', l_factorization, R)
    return result

def grad_mu(l_factorization, l_nonnegativity, l_relaxation,
                        mu, R, L, D, d):
    """
    Returns gradients of the Lagrange multipliers term of the Lagrangian, with
    respect to moments vector mu
    """
    result = np.zeros((L, D, 2*d + 1))

    # factorization infeasibilities, linear in mu for each
    # individual matrix term (one mu may occupy multiple terms in M_d)
    # this is following the cross-diagonal form of M_d,
    # that (M_d)_{a,b} = mu_{a+b}
    for n_i in range(2*d + 1):
        lower = max(0, n_i - d)
        number = (d+1) - abs(d - n_i)
        upper = lower + number
        for k in range(number):
            result[:,:,n_i] += l_factorization[:,:,lower+k,upper-1-k]

    # highlight infeasible mu_1,0 (in this case, negatives)
    infeas = np.copy(mu[:,0,0])
    infeas[infeas >= 0] = 0
    result[:,0,0] += np.where(infeas < 0, l_nonnegativity[:,0], infeas)

    # gradient for mu_i,0 for i = 2, ..., D (constraint is == 1)
    # this is just the Lagrange multiplier because constraint is linear
    # function of mu
    result[:,1:,0] += l_nonnegativity[:,1:]

    # gradient for the relaxation constraint from B.2.1, restricting absolute
    # values to <= 1
    # gradient is sign(mu) * Lagrange multiplier if |mu| > 1, 0 otherwise
    A = np.maximum(np.abs(mu[:,:,:d+1]) -
                   np.ones((L, D, d+1)), 0)
    signed_l_relaxation = np.sign(mu[:,:,:d+1]) * l_relaxation
    result[:,:,:d+1] += np.where(A > 0, signed_l_relaxation, A)

    return result

def grad_lm_fact(l_factorization, l_nonnegativity, l_relaxation,
                        mu, R, L, D, d):
    """
    Returns gradients of the Lagrange multipliers term of the Lagrangian, with
    respect to the factorization terms of the Lagrange multipliers vector
    """
    # RRt = R @ R.T
    RRt = jnp.einsum('abik,abjk->abij', R, R)
    # Set up M_d
    M_d = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
             for m in range(d+1)]
             for i in range(D)]
             for l in range(L)])

    # Each multiplier corresponds to one entry of this tensor
    return M_d - RRt

def grad_lm_nonnegativity(l_factorization, l_nonnegativity, l_relaxation,
                        mu, R, L, D, d):
    """
    Returns gradients of the Lagrange multipliers term of the Lagrangian, with
    respect to the factorization terms of the Lagrange multipliers vector
    """
    result = np.zeros((L, D))
    # 5. mu_(1,0)^l>=0, so linear if mu < 0, 0 otherwise
    result[:,0] += np.minimum(mu[:,0,0], 0)

    # 6. mu_(i,0)^l - 1 = 0 for i = 2, ..., D, so linear
    result[:,1:] += mu[:,1:,0].reshape(L, D-1) - np.ones((L, D-1))
    #jnp.einsum('ij,ij->',
    #              mu[:,1:,0].reshape(L, D-1) -
    #              jnp.ones((L, D-1)),
    #              l_nonnegativity[:,1:])
    return result

def grad_lm_relaxation(l_factorization, l_nonnegativity, l_relaxation,
                        mu, R, L, D, d):
    """
    Returns gradients of the Lagrange multipliers term of the Lagrangian, with
    respect to the factorization terms of the Lagrange multipliers vector
    """
    # returns zero for each component with mu feasible, and |mu| - 1 for every
    # component of mu in the infeasible i.e. |mu| > 1 region
    return np.maximum(np.abs(mu[:,:,:d+1]) -
                        np.ones((L, D, d+1)), 0)


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

def phi(n, mu, D, L):
    """
    Calculates the sum of the product measures of a monomial given by n
    This corresponds to phi in the paper

    arguments:
    n  -- the tuple of exponents
    mu -- the moment vectors for D measures for each L product measures
    D  -- the dimension of the hypercube
    L  -- the number of product measures
    """

    # Set up array for calculations
    # Each row corresponds to a product measure
    # The entries are the moments specified by n for each component measure
    A = jnp.array([
        [mu[l,i,n_i] for (i, n_i) in zip(range(D), n)]
        for l in range(L)])
                            
    # Multiply over the rows, then add up the resulting product measure values
    return jnp.sum(jnp.prod(A, axis=1))

def new_objective(coef, powers, M, D, L):
    """
    Calculates the objective function given the polynomial and the moment
    matrix

    arguments:
    coef -- a list of polynomial coefficients for each power tuple
    powers -- list of power tuples specifiying the monomial
    mu -- the moment vectors for product measures
    D -- the dimension of the hypercube
    L -- the number of product measures
    """
    # would numpy be faster, or would array creation slow it down?
    return sum([p_n * phi(n, mu, D, L) for p_n, n in zip(coef, powers)])

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


