import copy
from functools import partial
import itertools
import jax.numpy as jnp
from jax import grad as jaxgrad
#import numpy.lib.stride_tricks.as_strided as as_strided
# TODO fix strange thing with attempted import of as_strided alone
import numpy as np
import sympy as sp
from scipy.linalg import block_diag, hankel
from scipy.optimize import minimize
import torch
import itertools

# The support of the polynomial objective function
# coefficients - real-valued p_n for each monomial term
# powers - sequences n of multi-indexes for each monomial x^n
class PolySupport:
    def __init__(self, coefficients, powers):
        self.coefficients = coefficients
        self.powers = powers
        self.D = len(powers[0])
        self.d = max([max(n) for n in powers])

    def evaluate(self, x):
        total = 0
        for c, p in zip(self.coefficients, self.powers):
            term = 1
            for i in range(self.D):
                term *= (x[i]) ** p[i]

            total += c * term

        return total

class HessianComponent(PolySupport):
    """
    Object that represents one component of the Hessian of the objective
    function (moments version, not original polynomial version).
    Component is partial{objective}{del mu_{i, a} del mu_{j, b}},
    For creation, it is agnostic to total L and specific l, and to evaluate we
    pass it just a single mu^(l).
    """

    def __init__(self, objective, i, j, a, b):
        """
        arguments:
        objective -- PolySupport object of the objective function
        i -- index of component measure of "first" partial
        j -- index of component measure of "second" partial
        a -- degree of moment of mu_i
        b -- degree of moment of mu_j
        """

        self.D = objective.D
        self.d = objective.d

        self.i = i
        self.j = j
        self.a = a
        self.b = b

        # By construction of the problem, partial w/r/t same moment is always zero
        if i == j:
            self.zero = True
            return

        coef = []
        powers = []
        for obj_coef, obj_pow in zip(objective.coefficients, objective.powers):
            if obj_pow[i] == a and obj_pow[j] == b:
                coef.append(obj_coef)
                powers.append(obj_pow)

        self.coefficients = np.array(coef)
        self.powers = powers

        # if no terms contain both moments, indicate that this partial is
        # always zero
        self.zero = (len(powers) == 0)

    def evaluate(self, mu, l):
        """
        arguments:
        mu -- product measure mu to evaluate mixed partial with
        l -- which product measure mu^(l) to use
        """
        # return 0 if always 0
        if self.zero:
            return 0

        # Set up array for calculations
        # This uses n to signify a tuple of degrees, the paper's notation
        # Each row corresponds to a term phi_n(mu) in the sum 
        # The entries are the moments specified by n for each component measure
        A = np.array([
            [mu[l,k,n_k] for (k, n_k) in enumerate(n)]
            for n in self.powers])

        # Set the value of the moments we are differentiating away to 1
        A[:,self.i] = np.ones(len(self.powers))
        A[:,self.j] = np.ones(len(self.powers))
                                
        # Multiply over the rows (phi), then add up the values, multiplying each
        # by the relevant coefficient
        return np.prod(A, axis=1) @ self.coefficients

class Hessian():
    """
    Object for evaluating Hessians of a given objective polynomial at
    whatever moment vectors are passed to it.
    """

    def __init__(self, objective):
        """
        arguments:
        objective -- PolySupport object of the objective function
        """

        self.objective = objective

        self.D = objective.D
        self.d = objective.d
        
        # We construct the Hessian agnostic to L, and only use it later when
        # evaluating it given a moment tensor
        self.terms = [[[[HessianComponent(
            objective, i, j, a, b) for b in range(self.d+1)]
                                   for a in range(self.d+1)]
                                   for j in range(self.D)]
                                   for i in range(self.D)]

    def get_term(self, mu, l, i, j, a, b):
        return self.terms[i][j][a][b].evaluate(mu, l)

    def matrix(self, mu):
        L = mu.shape[0]                                           
        # TODO maybe change this indexing? Make it compatible with self.terms
        D = self.D
        d = self.d
        components = np.zeros((L, D, d+1, D, d+1))

        ranges = [
                range(L),
                range(D),
                range(d+1),
                range(D),
                range(d+1)]
        for l, i, a, j, b in itertools.product(*ranges):
            components[l, i, a, j, b] = self.get_term(mu, l, i, j, a, b)

        reshaped = np.reshape(components, (L, D*(d+1), D*(d+1)))
        return block_diag(*reshaped)

class ExampleF(PolySupport):
    """
    Generates the polynomial $f_D(x)$ from example 3.1 in Letourneau paper, in
    given dimension D
    """
    def __init__(self, D):
        # define variable x_1, x_2, ..., x_D
        x = sp.symbols(f'x1:{D+1}')
        
        # Create the polynomial
        T_2 = [sp.polys.orthopolys.chebyshevt_poly(2, x=x[i]) for i in range(D)]
        T_8 = [sp.polys.orthopolys.chebyshevt_poly(8, x=x[i]) for i in range(D)]
        product = 1
        for i in range(D):
            product *= T_8[i]
        polynomial = (1 / D) * sum(T_2) - product
      
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
            n_int = tuple(int(n_i) for n_i in n)
            powers.append(n_int)

        super().__init__(coefficients, powers)

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
            n_int = tuple(int(n_i) for n_i in n)
            powers.append(n_int)

        super().__init__(coefficients, powers)

class PlotPoly(PolySupport):
    """
    Generates the polynomial $g_D(x)$ from example 3.2 in Letourneau paper, in
    given dimension D
    """
    def __init__(self, D):
        # define variable x_1, x_2, ..., x_D
        x = sp.symbols(f'x1:{D+1}')

        powers = tuple(itertools.product((2, 4, 6), repeat=D))
        #for term in powers:
        #    print(term)
        a = [power.count(2) for power in powers]
        b = [power.count(4) for power in powers]
        c = [power.count(6) for power in powers]
        # check they all have same total of powers
        #for i in range(len(powers)):
        #    print(a[i] + b[i] + c[i])
        exponents = np.array((a, b, c)).T
        base = np.array([9 / 512.0, -25.0 / 256, 1.0 / 6])
        raised = np.power(base, exponents)
        #print(np.concatenate((exponents, raised), axis=1))
        coefficients = np.prod(raised, axis=1)
        coefficients = np.expand_dims(coefficients, axis=1)
        #print(np.concatenate((exponents, coefficients), axis=1))
        #print(coefficients)
        #print(powers)
        #print(coefficients.shape)

        super().__init__(coefficients, powers)

class PlotPolySum(PolySupport):
    """
    Generates the polynomial $g_D(x)$ from example 3.2 in Letourneau paper, in
    given dimension D
    """
    def __init__(self, D):
        # define variable x_1, x_2, ..., x_D
        x = sp.symbols(f'x1:{D+1}')

        #base = 100 * np.array([9 / 512.0, -25.0 / 256, 1.0 / 6])
        base = 10 * np.array([9 / 512.0, -25.0 / 256, 1.0 / 6])
        power_mask = np.array([2, 4, 6]).T
        powers = np.zeros((3*D, D), dtype=int)
        coefficients = np.zeros((3*D,))
        for i in range(D):
            powers[3*i:3*i+3,i] = power_mask
            coefficients[3*i:3*i+3] = base.T

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

        self.factorization = np.zeros((L, D, d+1, d+1))
        self.nonnegativity = np.zeros((L, D))
        self.relaxation = np.zeros((L, D, d+1))

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

    def update(self, free_vars, gamma):
        """
        Update Lagrange Multipliers according to the Burer Monteiro 2003 paper.
        Note that gamma here corresponds to sigma in their paper, and that
        we are adding to the multipliers when they subtract because they flip
        a sign in their definition of the augmented Lagrangian
        """
        free_vars.update_M_d()
        free_vars.update_RRt()
        self.factorization += gamma * (free_vars.M_d - free_vars.RRt)

        #self.nonnegativity[:,0] += gamma * np.minimum(free_vars.mu[:,0,0], 0)
        # NOTE changed
        self.nonnegativity += gamma * (free_vars.mu[:,:,0].reshape(self.L, self.D) -
                      np.ones((self.L, self.D)))
        self.relaxation += gamma * np.maximum(np.abs(free_vars.mu[:,:,:self.d+1]) -
                       np.ones((self.L, self.D, self.d+1)), 0)

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
class FreeVariables:
    def __init__(self, L, D, d, mu=None, R=None, seed=None):
        self.L = L
        self.D = D
        self.d = d

        random = np.random.default_rng(seed) # None will yield OS-selected seed

        self.mu = np.array(mu) if mu is not None else random.random(
                size=(L, D, 2 * d +1))
        self.M_d = np.array([[[[self.mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])

        if R is not None:
            self.R = np.array(R)
        else:
            random_R = random.random(size=(L, D, d+1, d+1)) * 2 - np.ones((L, D, d+1, d+1))
            self.R = random_R

        # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
        self.update_RRt()

    def flattened(self):
        # TODO does this copy the arrays? If not, does this cause problems?
        #return np.concatenate((self.mu, self.R), axis=0)
        return np.concatenate((self.mu.flatten(), self.R.flatten()), axis=0)

    def update_RRt(self):
        # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
        self.RRt = np.einsum('abik,abjk->abij', self.R, self.R)

    def update_M_d(self):
        #self.M_d = jnp.array([[[[self.mu[l,i,n+m] for n in range(d+1)]
        #         for m in range(d+1)]
        #         for i in range(D)]
        #         for l in range(L)])
        for n in range(self.d+1):
            for m in range(self.d+1):
                self.M_d[:,:,n,m] = self.mu[:,:,n+m]

    def optimal_location(self):
        # calculate probability masses of each product measure
        masses = np.prod(self.mu[:,:,0], axis=1)
        l = np.argmax(masses)
        x = np.zeros((self.D,))
        denominator = np.prod(self.mu[l,:,0])
        x = self.mu[l,:,1] / denominator
        #for i in range(self.D):
        #    x[i] = self.mu[l,i,1] / np.prod(self.mu[l,:,0]
        return x

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

def new_objective(mu, coef, powers, L, D):
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

def multiply_lagrangian(l_factorization, l_nonnegativity, l_relaxation,
                        mu, M_d, R, L, D, d):
    """
    Method to multiply lagrange multiplier vector by the infeasibilities.
    Separated out from the LagrangeMultipliers class for use by jax autogradient
    """
    # RRt = R @ R.T for each of the D x L factorizations M = R @ R.T
    RRt = jnp.einsum('abik,abjk->abij', R, R)
    total = 0

    # 1.Md(mu_0^(l)) - R_0^l R_0^l.T = 0
    # Penalize inaccurate factorizations
    # add up Lagrange multipliers times each componentwise difference
    # between M_d and R @ R.T
    total += jnp.einsum('abij,abij->', M_d - RRt, l_factorization)
    
    # NOTE changed from paper's >= 0 to == 1 (below)
    # so this part is now obsolete
    # 5. mu_(1,0)^l>=0, so anything positive is clipped
    #total += jnp.minimum(mu[:,0,0], 0) @ l_nonnegativity[:,0]

    # NOTE this is a change from the paper. Now we require all zeroth moments
    # in first coordinate to be 1
    # 6. mu_(i,0)^l - 1 = 0
    total += jnp.einsum('ij,ij->',
                  mu[:,:,0].reshape(L, D) -
                  jnp.ones((L, D)),
                  l_nonnegativity[:,:])

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

def print_new_penalty(mu, M_d, R, gamma, L, D, d):
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
    # DEBUGGING
    print('norm(M_d - RRt) = {}'.format(np.linalg.norm(diff.flatten(), ord=1)))
    total += jnp.einsum('abij,abij->', diff, diff)
    
    # 5. mu_(1,0)^l>=0, so anything positive is clipped
    negatives = jnp.minimum(mu[:,0,0], 0)
    # DEBUGGING
    #print('mu[:,0,0] = {}'.format(mu[:,0,0]))
    total += negatives @ negatives

    # 6. mu_(i,0)^l - 1 = 0
    diff = mu[:,1:,0].reshape(L, D-1) - jnp.ones((L, D-1))
    # DEBUGGING
    print('|mu-1| = {}'.format(np.linalg.norm(diff.flatten(), ord=1)))
    total += jnp.einsum('ij,ij->', diff, diff)

    # B.2.1. check that relevant moments have absolute value at most 1
    A = jnp.maximum(jnp.abs(mu[:,:,:d+1]) -
                   jnp.ones((L, D, d+1)), 0)
    total += jnp.einsum('ijk,ijk->', A, A)

    print('penalty = {}'.format((gamma / 2) * total))
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

    # NOTE this has been removed, in a digression from the paper
    # we have substituted in the thing below
    # nonnegativity infeasibilities, >= 0 and == 1 for i = 2, ..., D
    # highlight infeasible mu_1,0 (in this case, negatives)
    # TODO fix nondifferentiability issues with this for == 0
    #infeas = np.copy(mu[:,0,0])
    #infeas[infeas >= 0] = 0
    #result[:,0,0] += -1 * infeas

    # NOTE changed from paper to == 1 for all, not just i = 2, ..., D
    # gradient for mu_i,0 for i = 2, ..., D (constraint is == 1)
    result[:,:,0] += mu[:,:,0] - np.ones((L, D))

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
    return 2 * gamma * (RRt - M_d) @ R

def grad_lm_R(l_factorization, l_nonnegativity, l_relaxation,
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

    # NOTE removed because we're making it == 1 for all i = 1, ..., D
    # highlight infeasible mu_1,0 (in this case, negatives)
    #infeas = np.copy(mu[:,0,0])
    #infeas[infeas >= 0] = 0
    #result[:,0,0] += np.where(infeas < 0, l_nonnegativity[:,0], infeas)

    # This averages the "derivative" in both directions because of the
    # nondifferentiability of the infeasibility function at 0
    #result[:,0,0] += 0.5 * np.where(mu[:,0,0] < 0, l_nonnegativity[:,0], np.zeros_like(mu[:,0,0]))
    #result[:,0,0] += 0.5 * np.where(mu[:,0,0] <= 0, l_nonnegativity[:,0], np.zeros_like(mu[:,0,0]))

    # NOTE changed to all i = 1, ..., D from i = 2, ..., D
    # gradient for mu_i,0 for i = 1, ..., D (constraint is == 1)
    # this is just the Lagrange multiplier because constraint is linear
    # function of mu
    result[:,:,0] += l_nonnegativity[:,:]

    # gradient for the relaxation constraint from B.2.1, restricting absolute
    # values to <= 1
    # gradient is sign(mu) * Lagrange multiplier if |mu| > 1, 0 otherwise
    A = np.maximum(np.abs(mu[:,:,:d+1]) -
                   np.ones((L, D, d+1)), 0)
    absolute = np.abs(mu[:,:,:d+1])
    signed_l_relaxation = np.sign(mu[:,:,:d+1]) * l_relaxation

    # imitate auto-gradient by effectively averaging the gradients at the nondifferentiable
    # point at 1 (0 for |mu| < 1, signed lambda for |mu| > 1)
    avg_result = 0.5 * np.where(absolute >= 1, signed_l_relaxation, np.zeros_like(absolute))
    avg_result += 0.5 * np.where(absolute > 1, signed_l_relaxation, np.zeros_like(absolute))
    #result[:,:,:d+1] += np.where(absolute >= 1, signed_l_relaxation, np.zeros_like(absolute))
    result[:,:,:d+1] += avg_result

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
    #result = np.zeros((L, D))
    # NOTE this has been removed as we're now making it all == 1
    # 5. mu_(1,0)^l>=0, so linear if mu < 0, 0 otherwise
    #result[:,0] += np.minimum(mu[:,0,0], 0)

    # NOTE changed from i = 2, ..., D
    # see old commits for old code, just an index +1
    # 6. mu_(i,0)^l - 1 = 0 for i = 1, ..., D, so linear
    result = mu[:,:,0].reshape(L, D) - np.ones((L, D))
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

def grad_objective(mu, coef, powers, L, D, d):
    """
    Calculate the gradient of the objective polynomial with
    respect to each product measure term
    """
    # Although the last d moments don't matter, we need this shape to be
    # comparable with the jax result
    result = np.zeros((L, D, 2 * d+1))

    term = np.zeros((L, D))
    for p_n, n in zip(coef, powers):
        # Create L x D array of moments in the monomial
        A = np.array([mu[:,i,n_i] for (i, n_i) in zip(range(D), n)]).T

        # Stack up copies to use one for each of the D partials we will
        # calculate for each of the L layers
        B = np.stack([A for _ in range(D)], axis=0)

        # Set the mu term of which we're taking the partial to 1, which
        # sets up our multiplication below to yield exactly the partial
        for i in range(D):
            B[i,:,i] = np.ones((L,))

        B = np.prod(B, axis=2)

        # for each product measure term, add partial from each L in parallel
        for (i, n_i) in zip(range(D), n):
            result[:,i,n_i] += p_n * B[i,:]

    return result

def new_augmented_lagrangian(free_vars, lm, coef, powers, gamma, L, D, d):
    """
    free_vars - passed as a 1-D array for scipy's minimize function
    M_d - passed separately, is updated according to free_vars
    """
    # Reconstruct mu and R from flattened version without having to reshape them
    mu_size = L * D * (2 * d + 1)
    mu = np.lib.stride_tricks.as_strided(free_vars[:mu_size],
                    shape=(L, D, 2*d + 1),
                    writeable=False)
    R = np.lib.stride_tricks.as_strided(free_vars[mu_size:], shape=(L, D, d+1, d+1), writeable=False)
    M_d = np.zeros((L, D, d+1, d+1))
    for n in range(d+1):
        for m in range(d+1):
            M_d[:,:,n,m] = mu[:,:,n+m]

    return (new_objective(mu, coef, powers, L, D)
            + multiply_lagrangian(lm.factorization, lm.nonnegativity,
                                  lm.relaxation, mu, M_d, R, L, D, d)
            + new_penalty(mu, M_d, R, gamma, L, D, d))

def new_gradient(free_vars, lm, coef, powers, gamma, L, D, d):
    """
    Gradient of the new augmented lagrangian with respect to the free variables
    mu, R contained in free_vars (reflected in M_d too)
    """
    # Reconstruct mu and R from flattened version without having to reshape them
    mu_size = L * D * (2 * d + 1)
    #mu = np.lib.stride_tricks.as_strided(free_vars[:mu_size],
    #                shape=(L, D, 2*d + 1),
    #                writeable=False)
    mu = np.copy(free_vars[:mu_size]).reshape((L, D, 2*d + 1))

    #R = np.lib.stride_tricks.as_strided(free_vars[mu_size:],
    #                                    shape=(L, D, d+1, d+1),
    #                                    writeable=False)
    R = np.copy(free_vars[mu_size:]).reshape((L, D, d+1, d+1))

    M_d = np.zeros((L, D, d+1, d+1))
    for n in range(d+1):
        for m in range(d+1):
            M_d[:,:,n,m] = mu[:,:,n+m]

    mu_grad = np.zeros((L, D, 2*d + 1))
    mu_grad += grad_objective(mu, coef, powers, L, D, d)
    mu_grad += grad_mu(lm.factorization, lm.nonnegativity, lm.relaxation,
                       mu, R, L, D, d)
    mu_grad += grad_penalty_mu(mu, M_d, R, gamma, L, D, d)

    R_grad = grad_penalty_R(mu, M_d, R, gamma, L, D, d)
    R_grad += grad_lm_R(lm.factorization, lm.nonnegativity, lm.relaxation,
                       mu, R, L, D, d)

    # return gradients flattened and concatenated
    return np.concatenate((mu_grad.flatten(), R_grad.flatten()), axis=0)

#TODO update docstring
def solver(poly, L=6, max_iter=10, gamma=10, multiplier=10, eta=0.25,
           epsilon=1e-6, initial_mu=None, initial_R=None, seed=1243124242, verbose=True):
    """
    L is the number of measures
    rho is the value of penalty term gamma
    This function will output a global mimimum point of polynomial on
    [-1,1]^{D} and it's relative error subject to the real minimum value
    epsilon is for stopping condition
    Lack a good stop condition and time of running is too long
    """
    coef = poly.coefficients
    powers = poly.powers

    # extract dimension of hypercube by number of x_i variables in polynomial
    # extract highest degree of a single variable x_i in polynomial
    D = len(powers[0])
    powers_array = np.array(powers)
    d = int(np.max(powers_array)) # in weird case where we generated powers with numpy

    # TODO change how this is managed, may be best to use exclusively arrays
    # and not bother with this object
    #free_vars_obj = FreeVariables(L, D, d, seed=seed)
    free_vars_obj = FreeVariables(L, D, d, mu=initial_mu, R=initial_R)
    free_vars = free_vars_obj.flattened()
    M_d = free_vars_obj.M_d
    if verbose:
        print('Objective value / L = {}'.format(
            new_objective(free_vars_obj.mu, coef, powers, L, D) / L))

    lm = LagrangeMultipliers(L, D, d)

    if verbose:
        print("(L, D, d) = ({}, {}, {})".format(L, D, d))

    # v_k is the penalty term not scaled by gamma / 2
    v_k = (2 / gamma) * new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                                    free_vars_obj.R, gamma, L, D, d)

    # Initial x location
    x_min = free_vars_obj.optimal_location()
    if verbose:
        print('Initial x location = {}'.format(x_min))
        print_new_penalty(free_vars_obj.mu, free_vars_obj.M_d, free_vars_obj.M_d,
                          gamma, L, D, d)

    cur_obj = 1e8 # start out objective at very high value

    for iteration in range(max_iter):
        # NOT a partial derivative
        partial_func = partial(new_augmented_lagrangian, lm=lm,
                               coef=coef, powers=powers, gamma=gamma, L=L, D=D,
                               d=d)
        # NOT a partial derivative
        partial_grad = partial(new_gradient, lm=lm,
                               coef=coef, powers=powers, gamma=gamma, L=L, D=D,
                               d=d)

        result = minimize(partial_func, x0=free_vars,
                        method='L-BFGS-B',
                        jac=partial_grad,
                        options={
                            #'gtol': 1e-5,             # Stopping criterion (relative gradient)
                            #'ftol': 1e-7,             # Stopping criterion (absolute value)
                            # better tols
                            #'gtol': 1e-6,             # Stopping criterion (relative gradient)
                            #'ftol': 1e-9,             # Stopping criterion (absolute value)
                            # best tols
                            'gtol': 1e-7,             # Stopping criterion (relative gradient)
                            'ftol': 1e-11,             # Stopping criterion (absolute value)
                            'maxcor': 40,             # The order of the approximation Hessian
                        })
        
        if verbose:
            print("\nIteration: {}".format(iteration))
            print("min L = ", result.fun)
            #print("Was the optimization successful?", result.success)
            print("Number of L-BFGS iterations:", result.nit)
            #print(result.message)
        print("Number of L-BFGS iterations:", result.nit)

        # update free variables and our object tracking them
        old_free_vars = np.copy(free_vars)
        free_vars = np.copy(result.x)
        mu_size = L * D * (2 * d + 1)
        free_vars_obj.mu = np.reshape(np.copy(free_vars[:mu_size]), (L, D, 2*d+1))
        free_vars_obj.R = np.reshape(np.copy(free_vars[mu_size:]), (L, D, d+1, d+1))
        free_vars_obj.update_M_d()
        prev_obj = cur_obj
        cur_obj = new_objective(free_vars_obj.mu, coef, powers, L, D) / L
        print('Objective value / L = {}'.format(
            new_objective(free_vars_obj.mu, coef, powers, L, D) / L))
        if verbose:
            print('Objective value / L = {}'.format(
                new_objective(free_vars_obj.mu, coef, powers, L, D) / L))
        
        if verbose:
            print_new_penalty(free_vars_obj.mu, free_vars_obj.M_d, free_vars_obj.R,
                              gamma, L, D, d)

        # Update lm or gamma according to BM paper (note our gamma is their sigma)
        v = (2 / gamma ) * new_penalty(free_vars_obj.mu, free_vars_obj.M_d,
                                       free_vars_obj.R, gamma, L, D, d)
        if verbose:
            print('v = {}'.format(v))

        if v < eta * v_k:
            lm.update(free_vars_obj, gamma)
            v_k = v
            if verbose:
                print('updated lagrangian')
        else:
            gamma *= multiplier
            if verbose:
                print('updated gamma = {}'.format(gamma))

        if verbose:
            print('v_k = {}'.format(v_k))

        # Calculate the x_min
        x_min = free_vars_obj.optimal_location()
        if verbose:
            print('current recovered minimizer = {}'.format(x_min))

        # break if feasible enough and objective hasn't moved much
        #if (np.linalg.norm(partial_grad(free_vars)) / (L*D*d*d) < 1e-1 and v_k < 1e-8 and
        #    np.abs(cur_obj - prev_obj) < epsilon):
        # remove gradient one for now
        if (True and v_k < 1e-8 and
            np.abs(cur_obj - prev_obj) < epsilon):
            print('D = {} breaking out of loop'.format(D))
            #if verbose:
                #print('breaking out of loop')
            break

    x_min = free_vars_obj.optimal_location()
    if verbose:
        print('final minimizer = {}'.format(x_min))
        print('mu = {}'.format(free_vars_obj.mu))
        #np.save('mu.npy', free_vars_obj.mu)

    print('number of iterations = {}'.format(iteration))
    np.save('mu_{}.npy'.format(D), free_vars_obj.mu)
    return (x_min, cur_obj)

def construct_matrix(mu):
    """
    Make sure mu is truly 1-D when passed to this
    """
    d = int(np.floor(mu.shape[0] / 2))
    c = mu[:d+1]
    r = mu[d:]
    M_d = hankel(c, r=r)
    return M_d

def test_psd(matrix, epsilon = 1e-3):
    evalues, evectors = np.linalg.eigh(matrix) 
    return np.all(evalues >= -1 * epsilon)

def test_feasible(mu, epsilon = 1e-3):
    if np.abs(mu[0] - 1) > epsilon:
        return False

    d = int(np.floor(mu.shape[0] / 2))
    ones = np.ones(d)
    if np.any(np.abs(mu[:d]) - ones > epsilon):
        return False

    return test_psd(construct_matrix(mu))

def test_feasible_direction(mu, perturbation, max_iter=4, epsilon = 1e-3):
    """
    Normalizes perturbation direction, returns largest power of 10 t for
    which mu + t * v is still feasible
    """
    if not test_feasible(mu, epsilon):
        print('given matrix is not feasible')
        return

    v = perturbation / np.linalg.norm(perturbation)
    t = 1
    for i in range(max_iter):
        if test_feasible(mu + t * v, epsilon):
            return (True, t)
        t = t / 10
    
    return (False, t)

def project_hankel(matrix):
    """
    Returns a copy of square matrix projected onto space of Hankel matrices
    """
    n = matrix.shape[0]
    copy = np.copy(matrix)
    flipped = np.fliplr(copy)
    for k in range(-n+1, n):
        antidiagonal = np.diagonal(flipped, offset=k)
        mean = np.mean(antidiagonal)
        for i in range(n):
            j = i + k
            if 0 <= j and j < n:
                flipped[i,j] = mean

    return copy

def project_C_1(matrix):
    """
    Projects the matrix onto the space of Hankel matrices with
    entries between -1 and 1 and mu_0 == 1
    """
    proj = project_hankel(matrix)
    proj[0,0] = 1
    np.clip(proj, a_min=-1, a_max=1, out=proj)
    return proj

def project_C_2(matrix):
    """
    Project the symmetric matrix onto the PSD cone
    """
    evalues, evectors = np.linalg.eigh(matrix)
    proj = np.zeros_like(matrix)
    for lm, v in zip(evalues, evectors.T):
        if lm > 0:
            proj += lm * np.outer(v, v)

    return proj


def dykstra(matrix, f=project_C_1, g=project_C_2, max_iter=1_000, epsilon=1e-3):
    """
    Calculate the projection of matrix onto the intersection of two convex sets C_1, C_2,
    given functions f and g which project a symmetric matrix onto them respectively.
    """
    h_t = matrix
    p_t = np.zeros_like(matrix)
    q_t = np.zeros_like(matrix)
    for i in range(max_iter):
        y_t = f(h_t + p_t)
        h_next = g(y_t + q_t)
        p_t += h_t - y_t
        q_t += y_t - h_next

        if (np.linalg.norm(y_t - h_t, ord='fro') < epsilon and
            np.linalg.norm(y_t - h_next, ord='fro') < epsilon):
            print('iteration = {}'.format(i))
            return y_t

        h_t = h_next

    print('went beyond max_iter')
    return y_t

def non_psd_perturbation(matrix, epsilon = 1e-3):
    """
    Return a matrix whose sum with the given matrix (assumed PSD) is not PSD
    """
    evalues, evectors = np.linalg.eigh(matrix) 
    if np.min(evalues) > epsilon:
        raise ValueError('matrix has no 0 eigenvalues i.e. is in interior of PSD cone')
    elif np.min(evalues) < -1 * epsilon:
        raise ValueError('matrix has negative eigenvalues i.e. is not in PSD cone')

    perturbation = np.zeros_like(matrix)
    for lm, v in zip(evalues, evectors.T):
        if np.abs(lm) < epsilon:
            normalized = v / np.linalg.norm(v)
            perturbation += -1 * np.outer(normalized, normalized)

    return perturbation

def ortho_hankel(d, k):
    mu = np.zeros(2*d+1)
    if k <= d:
        i = k+1
    else:
        i = 2*d +1 - k
    mu[k] = 1 / np.sqrt(i)
    return construct_matrix(mu)
