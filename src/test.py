import jax
import jax.numpy as jnp
import numpy as np
import unittest
from utils import *

class TestPhi(unittest.TestCase):
    def test_0(self):
        L = 2
        D = 1
        d = 6

        mu_vector = jnp.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = jnp.array([[jnp.copy(mu_vector) for d in range(D)] for l in range(L)])

        for i in range(d+1):
            n = (i,)
            truth = 2 / (i+1) if i % 2 == 0 else 0
            self.assertEqual(phi(n, mu, D, L), truth)

class TestPolynomial(unittest.TestCase):
    def setUp(self):
        return

    def test_0(self):
        p = ExampleG(2)
        true_coefficients = [4.00000000000000,
                             1/8,
                             3/8,
                             -4.00000000000000,
                             3/8,
                             4.00000000000000,
                             1/8,
                             -4,
                             1.00000000000000]
        true_powers = [(4, 0),
                       (3, 0),
                       (2, 1),
                       (2, 0),
                       (1, 2),
                       (0, 4),
                       (0, 3),
                       (0, 2),
                       (0, 0)]
        self.assertEqual(p.coefficients, true_coefficients)
        self.assertEqual(p.powers, true_powers)

    def test_1(self):
        p = ExampleG(3)
        true_coefficients = [8 / 3,
                             1 / 27,
                             1 / 9,
                             1 / 9,
                             -8 / 3,
                             1 / 9,
                             2 / 9,
                             1 / 9,
                             8 / 3,
                             1 / 27,
                             1 / 9,
                             -8 / 3,
                             1 / 9,
                             8 / 3,
                             1 / 27,
                             -8 / 3,
                             1.0]
        true_powers = [(4, 0, 0),
                       (3, 0, 0),
                       (2, 1, 0),
                       (2, 0, 1),
                       (2, 0, 0),
                       (1, 2, 0),
                       (1, 1, 1),
                       (1, 0, 2),
                       (0, 4, 0),
                       (0, 3, 0),
                       (0, 2, 1),
                       (0, 2, 0),
                       (0, 1, 2),
                       (0, 0, 4),
                       (0, 0, 3),
                       (0, 0, 2),
                       (0, 0, 0)]
        self.assertEqual(p.coefficients, true_coefficients)
        self.assertEqual(p.powers, true_powers)

class TestLagrangeMultipliers(unittest.TestCase):
    def setUp(self):
        L = 2
        D = 2
        d = 4

        self.L = L
        self.D = D
        self.d = d

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = jnp.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = jnp.array([[jnp.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = jnp.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = jnp.zeros(M.shape)
        RRt = jnp.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)
        # lambda
        self.lm = LagrangeMultipliers(L, D, d)


    def test_factorization_penalty(self):
        # Test that the lm.multiply() function properly evaluates the
        # lagrangian when multipliers are adjusted to align with various
        # violations of constraints (R = 0 initially)
        self.assertEqual(self.lm.multiply(self.free_vars), 0)

        self.lm.factorization[0,0,3,1] = 1
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        # Should add nothing to product, no violation here
        self.lm.factorization[0,0,2,1] = 1
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        self.lm.factorization[0,0,4,2] = -1
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 1/5 - 1/7)

        # Fix violation in M_d[0,0,3,1]
        self.free_vars.R[0,0,3,0] = 1 / 5
        self.free_vars.R[0,0,1,0] = 1
        self.free_vars.update_RRt()
        self.assertEqual(self.lm.multiply(self.free_vars), -1/7)


    def test_nonnegativity_penalty(self):
        self.lm.nonnegativity[1,0] = 3.7
        self.free_vars.mu[1,0,0] = -1.9
        self.assertEqual(self.lm.multiply(self.free_vars), -7.03)

    def test_mu_equality_constraints(self):
        self.lm.nonnegativity[1,1] = 5.8
        self.free_vars.mu[1,1,0] = 8.03
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        # should not change value
        self.lm.nonnegativity[0,1] = -1_000
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        self.free_vars.mu[0,1,0] = 0.5
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 540.774)

    # TODO test moments absolute value <= 1 constraints
    # TODO redundant B.2.2 numerical stability constraint

class TestObjectiveGradient(unittest.TestCase):
    def setUp(self):
        L = 2
        D = 2
        d = 4

        self.L = L
        self.D = D
        self.d = d

        self.p = ExampleG(D)

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        self.mu = np.array([[np.copy(mu_vector) for i in range(D)] for l in range(L)])

        # gradient with respect to moments and moment matrices
        self.jax_grad = jaxgrad(partial(new_objective,
                                        coef=self.p.coefficients,
                                        powers=self.p.powers,
                                        L=L, D=D),
                                argnums=(0,))

        def auto_grad(mu):
            #return np.copy(self.jax_grad(mu)[0])
            return self.jax_grad(mu)[0]

        self.grad = auto_grad

        # old gradient with respect to moments and moment matrices
        self.old_jax_grad = jaxgrad(partial(term_1,
                                            coefficients_list=self.p.coefficients,
                                            orders_list=self.p.powers,),
                                argnums=(2,))

        def old_auto_grad(mu):
            transposed = np.transpose(mu, axes=(1, 0, 2))
            old = self.old_jax_grad(D, L, transposed)[0]
            return np.transpose(old, axes=(1, 0, 2))

        self.old_grad = old_auto_grad

    def test_new(self):
        """
        Test gradient of objective function
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the uniform mu gives the correct answer
        jax_result = self.grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if random mu gives correct answer
        self.mu = 2 * self.rand.random_sample((L, D, 2*d + 1)) - 1
        jax_result = self.grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

    def test_old(self):
        """
        Test gradient of objective function compared to old autogradient
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the uniform mu gives the correct answer
        old_result = self.old_grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())

        # Test if random mu gives correct answer
        self.mu = 2 * self.rand.random_sample((L, D, 2*d + 1)) - 1
        old_result = self.old_grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())


class TestMultiplierGradient(unittest.TestCase):
    def setUp(self):
        L = 2
        D = 2
        d = 4

        self.L = L
        self.D = D
        self.d = d

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = np.zeros(M.shape)
        RRt = np.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

        # gradient with respect to Lagrange multipliers
        self.grad_lm = jaxgrad(partial(multiply_lagrangian), argnums=(0, 1, 2))

        # gradient with respect to moments and moment matrices
        self.jax_grad_mu = jaxgrad(partial(multiply_lagrangian, L=L, D=D, d=d), argnums=(3, 4))
        # define function to extract raw jax autogradient output and process it to
        # account for the relationship between mu and M_d
        def auto_grad_mu(l_factorization, l_nonnegativity, l_relaxation,
                                mu, M_d, R, L, D, d):
            jax_grad = self.jax_grad_mu(l_factorization, l_nonnegativity,
                                  l_relaxation, mu,
                                  M_d, R)
            j_mu = np.copy(jax_grad[0])
            j_M_d = np.copy(jax_grad[1])

            #grad_mu = np.zeros(L, D, 2 * d + 1)
            np_mu = np.copy(jax_grad[0])
            #grad_M_d = np.zeros((L, D, d+1, d+1))

            # factorization infeasibilities, linear in mu for each
            # individual matrix term (one mu may occupy multiple terms in M_d)
            # this is following the cross-diagonal form of M_d,
            # that (M_d)_{a,b} = mu_{a+b}
            for n_i in range(2*d + 1):
                lower = max(0, n_i - d)
                number = (d+1) - abs(d - n_i)
                upper = lower + number
                for k in range(number):
                    np_mu[:,:,n_i] += j_M_d[:,:,lower+k,upper-1-k]
                    #result[:,:,n_i] += l_factorization[:,:,lower+k,upper-1-k]

            return np_mu

        self.grad_mu = auto_grad_mu

        # gradient with respect to R
        self.grad_R = jaxgrad(partial(multiply_lagrangian, L=L, D=D, d=d), argnums=(5,))

        # wrapper function for old Lagrangian multiplier term, which requires us
        # to concatenate lagrange multipliers
        def old_multiply(mu, R, lm, L, D, d):
            old_mu = np.transpose(mu, axes=(1,0,2))
            old_R = np.transpose(R, axes=(1,0,2,3))

            old_factorization = np.transpose(lm.factorization, axes=(1, 0, 2, 3))
            old_nonnegativity = np.transpose(lm.nonnegativity, axes=(1, 0))
            old_relaxation = np.transpose(lm.relaxation, axes=(1, 0, 2))
            old_lm = [old_factorization, old_nonnegativity, old_relaxation]
            return term_2(D, L, old_mu, old_R, old_lm)


        # old gradient with respect to moments and factorization
        self.old_jax_grad = jaxgrad(partial(term_2),
                                argnums=(2, 3))

        def old_grad_mu(mu, R, lm, L, D, d):
            old_mu = np.transpose(mu, axes=(1,0,2))
            old_R = np.transpose(R, axes=(1,0,2,3))

            old_factorization = np.transpose(lm.factorization, axes=(1, 0, 2, 3))
            old_nonnegativity = np.transpose(lm.nonnegativity, axes=(1, 0))
            old_relaxation = np.transpose(lm.relaxation, axes=(1, 0, 2))
            old_lm = [old_factorization, old_nonnegativity, old_relaxation]
            old = self.old_jax_grad(D, L, old_mu, old_R, old_lm)[0].reshape((D, L, 2*d+1))
            return np.transpose(old, axes=(1, 0, 2))

        def old_grad_R(mu, R, lm, L, D, d):
            old_mu = np.transpose(mu, axes=(1,0,2))
            old_R = np.transpose(R, axes=(1,0,2,3))

            old_factorization = np.transpose(lm.factorization, axes=(1, 0, 2, 3))
            old_nonnegativity = np.transpose(lm.nonnegativity, axes=(1, 0))
            old_relaxation = np.transpose(lm.relaxation, axes=(1, 0, 2))
            old_lm = [old_factorization, old_nonnegativity, old_relaxation]
            old = self.old_jax_grad(D, L, old_mu, old_R, old_lm)[1].reshape((D, L, d+1, d+1))
            return np.transpose(old, axes=(1, 0, 2, 3))

        self.old_grad_mu = old_grad_mu
        self.old_grad_R = old_grad_R

    #@unittest.skip('incomplete')
    def test_mu(self):
        """
        Test gradient with respect to mu, incorporating all types of infeasibilities.
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if one lagrange multiplier factor and one R element change works
        self.lm.factorization[0,1,1,3] = 0.5
        self.free_vars.R[0,1,1,1] = 1
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if "random" (but fixed) M_d, factorization, and lagrange
        # multipliers yield the same answer
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d+1))
        self.free_vars.M_d = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

    def test_factorization_R(self):
        """
        Test gradient of factorization constraint with respect to factorization
        matrix R
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if one lagrange multiplier factor and one R element change works
        self.lm.factorization[0,1,1,3] = 0.5
        self.free_vars.R[0,1,1,1] = 1

        self.free_vars.update_M_d()
        self.free_vars.update_RRt()

        jax_result = self.grad_R(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R)
        old_result = self.old_grad_R(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_lm_R(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)

        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())

        # Test if "random" (but fixed) factorization and lagrange multipliers
        # yield the same answer
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))

        self.free_vars.update_M_d()
        self.free_vars.update_RRt()

        jax_result = self.grad_R(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R)
        old_result = self.old_grad_R(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_lm_R(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #print('mu = {}'.format(self.free_vars.mu))
        #print('R = {}'.format(self.free_vars.R))
        #print('lm_factorization = {}'.format(self.lm.factorization))

        #print('new_jax_grad = {}'.format(jax_result))
        #print('hardcode_grad = {}'.format(hardcoded_result))
        #print('old_jax_grad = {}'.format(old_result))

        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())

    def test_factorization_mu(self):
        """
        Test gradient of factorization constraint with respect to moment
        matrices determined by mu
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if one lagrange multiplier factor and one R element change works
        self.lm.factorization[0,1,1,3] = 0.5
        self.free_vars.R[0,1,1,1] = 1
        self.free_vars.update_M_d()
        self.free_vars.update_RRt()
        jax_result = self.grad_R(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R)
        old_result = self.old_grad_R(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_lm_R(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        print('mu = {}'.format(self.free_vars.mu))
        print('R = {}'.format(self.free_vars.R))
        print('lm_factorization = {}'.format(self.lm.factorization))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())

        print('jax = {}'.format(jax_result))
        print('hardcode = {}'.format(hardcoded_result))
        print('old_jax = {}'.format(old_result))

        # Test if "random" (but fixed) mu, factorization, and lagrange
        # multipliers yield the same answer
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d+1))

        self.free_vars.update_M_d()
        self.free_vars.update_RRt()

        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        print('mu = {}'.format(self.free_vars.mu))
        print('R = {}'.format(self.free_vars.R))
        print('lm_factorization = {}'.format(self.lm.factorization))

        print('jax = {}'.format(jax_result))
        print('hardcode = {}'.format(hardcoded_result))
        print('old_jax = {}'.format(old_result))

        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())

    def test_nonnegativity_mu(self):
        """
        Test the nonnegativity constraints' effect on the gradient of the
        Lagrange Multipliers term with respect to mu
        """
        L = self.L
        D = self.D
        d = self.d

        # Activate nonnegativity constraint
        self.lm.nonnegativity = self.rand.random_sample((L, D))
        self.free_vars.mu = np.array([[[-0.41620035, 0, 0, 0, 0, 0, 0, 0, 0],
          [-1.80757967, 0, 0, 0, 0, 0, 0, 0, 0]],
         [[ 0.97183684, 0, 0, 0, 0, 0, 0, 0, 0],
          [-0.81413875, 0, 0, 0, 0, 0, 0, 0, 0]]])
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        #self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        #self.assertTrue(np.isclose(old_result, jax_result).all())

        # Test if "random" (but fixed) factorization and lagrange multipliers
        # yield the same answer
        self.lm.nonnegativity = self.rand.random_sample((L, D))
        self.free_vars.mu = 5 * self.rand.random_sample((L, D, 2*d + 1)) - 4

        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        #self.assertTrue(np.isclose(old_result, jax_result).all())
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

    def test_relaxation_mu(self):
        """
        Test gradient of relaxation absolute value constraint with respect to mu.
        For info on constraint, see (B.2.1) in Letourneau
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if mu just outside feasible region and random lm works
        # NOTE because | mu - 1| is non-differentiable at mu = 1, we would
        # get discrepancies there because jax just takes the average of the
        # "derivative" in each direction, positive and negative, while the
        # hardcoded gradient returns 0
        self.free_vars.mu = np.ones((L, D, 2*d + 1))
        self.free_vars.M_d = self.rand.random_sample((L, D, d+1, d+1))
        self.lm.relaxation = self.rand.random_sample((L, D, d+1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())

        # Test if "random" (but fixed) mu, lagrange multipliers yield the same
        # answer
        self.free_vars.mu = 4 * self.rand.random_sample((L, D, 2*d + 1)) - 2
        self.lm.relaxation = self.rand.random_sample((L, D, d+1)) - 1
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.lm, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #print()
        #print('mu = {}'.format(self.free_vars.mu))
        #print('lm.relax = {}'.format(self.lm.relaxation))
        #print('jax = {}'.format(jax_result))
        #print('hardcode = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())

    def test_factorization_lm(self):
        """
        Test gradient of Lagrange multipliers term with respect to the
        Lagrange multipliers for the factorization infeasibilities
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if change in R works
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[0]
        hardcoded_result = grad_lm_fact(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Changing mu without updating M_d should break the agreement
        # between the jax gradient (uses mu and M_d) and our hardcoded one,
        # which uses only mu
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[0]
        hardcoded_result = grad_lm_fact(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertFalse(np.isclose(jax_result, hardcoded_result).all())

        # Updating M_d to reflect mu should fix the discrepancy from the test
        # above, and jax and our gradient should agree again
        self.free_vars.M_d = np.array([[[[self.free_vars.mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[0]
        hardcoded_result = grad_lm_fact(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #print()
        #print('mu = {}'.format(self.free_vars.mu))
        #print('lm.factorization = {}'.format(self.lm.factorization))
        #print('jax = {}'.format(jax_result))
        #print('hardcode = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        
    def test_nonnegativity_lm(self):
        """
        Test gradient of Lagrange multipliers term with respect to the
        Lagrange multipliers for the nonnegativity infeasibilities
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if changes in lm with no infeasibilities in mu does anything
        self.lm.nonnegativity = self.rand.random_sample((L, D))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[1]
        hardcoded_result = grad_lm_nonnegativity(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if changes in lm with plenty of infeasibilities in mu does anything
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d + 1))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[1]
        hardcoded_result = grad_lm_nonnegativity(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Changing mu without updating M_d should break the agreement
        # between the jax gradient (uses mu and M_d) and our hardcoded one,
        # which uses only mu
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[0]
        hardcoded_result = grad_lm_fact(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertFalse(np.isclose(jax_result, hardcoded_result).all())

        # Updating M_d to reflect mu should fix the discrepancy from the test
        # above, and jax and our gradient should agree again
        self.free_vars.M_d = np.array([[[[self.free_vars.mu[l,i,n+m] for n in range(d+1)]
                 for m in range(d+1)]
                 for i in range(D)]
                 for l in range(L)])
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[0]
        hardcoded_result = grad_lm_fact(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #print()
        #print('mu = {}'.format(self.free_vars.mu))
        #print('lm.factorization = {}'.format(self.lm.factorization))
        #print('jax = {}'.format(jax_result))
        #print('hardcode = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

    def test_relaxation_lm(self):
        """
        Test the gradient of Lagrange multipliers term with respect to the
        Lagrange multipliers for the relaxation infeasibilities |mu| <= 1
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if changes in lm with no infeasibilities correctly yields zeros
        self.lm.relaxation = self.rand.random_sample((L, D, d+1))
        all_zeros = np.zeros((L, D, d+1))
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[2]
        hardcoded_result = grad_lm_relaxation(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, all_zeros).all())
        self.assertTrue(np.isclose(hardcoded_result, all_zeros).all())

        # Test if adding infeasibilities does anything
        self.free_vars.mu = 4 * self.rand.random_sample((L, D, 2*d + 1)) - 2
        jax_result = self.grad_lm(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)[2]
        hardcoded_result = grad_lm_relaxation(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        #print()
        #print('mu = {}'.format(self.free_vars.mu))
        #print('lm.relaxation = {}'.format(self.lm.relaxation))
        #print('jax = {}'.format(jax_result))
        #print('hardcode = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        # There should be some infeasibilities here
        self.assertFalse(np.isclose(jax_result, all_zeros).all())
        self.assertFalse(np.isclose(hardcoded_result, all_zeros).all())

class TestPenaltyGradient(unittest.TestCase):
    def setUp(self):
        L = 2
        D = 2
        d = 4
        gamma = 10

        self.L = L
        self.D = D
        self.d = d
        self.gamma = gamma

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = np.zeros(M.shape)
        RRt = np.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

        # gradient with respect to moments and moment matrices
        self.jax_grad_mu = jaxgrad(
                partial(new_penalty, gamma=gamma, L=L, D=D, d=d), argnums=(0, 1, 2))
        # define function to extract raw jax autogradient output and process it to
        # account for the relationship between mu and M_d
        def auto_grad(mu, R):
            d = int(len(mu[0][0]) / 2)
            L = mu.shape[0]
            D = mu.shape[1]
            M_d = np.zeros((L, D, d+1, d+1))
            for n in range(d+1):
                for m in range(d+1):
                    M_d[:,:,n,m] = mu[:,:,n+m]
            jax_grad = self.jax_grad_mu(mu, M_d, R)
            j_mu = np.copy(jax_grad[0])
            j_M_d = np.copy(jax_grad[1])

            np_mu = np.copy(jax_grad[0])

            # factorization infeasibilities, linear in mu for each
            # individual matrix term (one mu may occupy multiple terms in M_d)
            # this is following the cross-diagonal form of M_d,
            # that (M_d)_{a,b} = mu_{a+b}
            for n_i in range(2*d + 1):
                lower = max(0, n_i - d)
                number = (d+1) - abs(d - n_i)
                upper = lower + number
                for k in range(number):
                    np_mu[:,:,n_i] += j_M_d[:,:,lower+k,upper-1-k]
                    #result[:,:,n_i] += l_factorization[:,:,lower+k,upper-1-k]

            return (np_mu, np.copy(jax_grad[2]))

        def auto_grad_mu(mu, R): return auto_grad(mu, R)[0]

        def auto_grad_R(mu, R): return auto_grad(mu, R)[1]

        self.grad_mu = auto_grad_mu
        self.grad_R = auto_grad_R

        #def old_penalty(mu, R, gamma, L, D, d):
        #    old_mu = np.transpose(mu, axes=(1,0,2))
        #    old_R = np.transpose(R, axes=(1,0,2,3))

        #    # gamma / 2 not included in function itself for some reason
        #    return (gamma / 2) * term_3(D, L, old_mu, old_R)


        ## old gradient with respect to moments and factorization
        #self.old_grad = jaxgrad(partial(old_penalty),
        #                        argnums=(0, 1))
        self.old_grad = jaxgrad(partial(term_3),
                                argnums=(2, 3))

        def old_grad_mu(mu, R, gamma, L, D, d):
            old_mu = np.transpose(mu, axes=(1,0,2))
            old_R = np.transpose(R, axes=(1,0,2,3))

            old = self.old_grad(D, L, old_mu, old_R)[0].reshape((D, L, 2*d+1))
            # multiply back gamma / 2 because it's not in original term_3
            return (gamma / 2) * np.transpose(old, axes=(1, 0, 2))

        def old_grad_R(mu, R, L, D, d):
            old_mu = np.transpose(mu, axes=(1,0,2))
            old_R = np.transpose(R, axes=(1,0,2,3))

            old = self.old_grad(D, L, old_mu, old_R)[1].reshape((D, L, d+1, d+1))
            return (gamma / 2) * np.transpose(old, axes=(1, 0, 2, 3))

        self.old_grad_mu = old_grad_mu
        self.old_grad_R = old_grad_R

    def test_mu(self):
        """
        Test gradient of penalty term with respect to mu
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the existing factorization gap M_d - R @ R.T is registered
        jax_result = self.grad_mu(self.free_vars.mu, self.free_vars.R)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.gamma, L, D, d)
        hardcoded_result = grad_penalty_mu(self.free_vars.mu,
                                           self.free_vars.M_d,
                                           self.free_vars.R, self.gamma, L, D,
                                           d)
        #print('gamma = {}'.format(self.gamma))
        #print('M_d = {}'.format(self.free_vars.M_d))
        #print('R @ R.T = {}'.format(self.free_vars.R @ self.free_vars.R))
        #print('jax_result = {}'.format(jax_result))
        #print('hardcoded_result = {}'.format(hardcoded_result))
        #print('old_result = {}'.format(old_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(hardcoded_result, old_result).all())

        # Test if "random" (but fixed) mu, M_d, and R give the same answer
        # Unlike for the gradients of the multiplier term, here there is no
        # interaction between entries of mu and M_d, so the answers will still
        # be the same even if they have no connection
        self.free_vars.mu = 2 * self.rand.random_sample((L, D, 2 * d+1)) - 1
        self.free_vars.update_M_d()
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.update_RRt()
        jax_result = self.grad_mu(self.free_vars.mu, self.free_vars.R)
        old_result = self.old_grad_mu(self.free_vars.mu, self.free_vars.R,
                                      self.gamma, L, D, d)
        hardcoded_result = grad_penalty_mu(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R, self.gamma, L, D, d)
        print('random test')
        print('gamma = {}'.format(self.gamma))
        print('mu = {}'.format(self.free_vars.mu))
        #print('M_d = {}'.format(self.free_vars.M_d))
        print('R @ R.T = {}'.format(self.free_vars.R @ self.free_vars.R))
        #print('jax_result = {}'.format(jax_result))
        print('hardcoded_result = {}'.format(hardcoded_result))
        print('old_result = {}'.format(old_result))
        print('diff = {}'.format(hardcoded_result - old_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(jax_result, old_result).all())
        self.assertTrue(np.isclose(hardcoded_result, old_result).all())

    def test_R(self):
        """
        Test gradient of factorization constraint with respect to R
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the existing factorization gap M_d - R @ R.T is registered
        # TODO add wrapper function to fix [0] unpacking for jax gradient
        jax_result = self.grad_R(self.free_vars.mu, self.free_vars.R)
        old_result = self.old_grad_R(self.free_vars.mu, self.free_vars.R,
                                      L, D, d)
        hardcoded_result = grad_penalty_R(self.free_vars.mu,
                                           self.free_vars.M_d,
                                           self.free_vars.R, self.gamma, L, D,
                                           d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())

        # Test if random R gives correct answer
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.update_M_d()
        self.free_vars.update_RRt()
        jax_result = self.grad_R(self.free_vars.mu, self.free_vars.R)
        old_result = self.old_grad_R(self.free_vars.mu, self.free_vars.R,
                                      L, D, d)
        hardcoded_result = grad_penalty_R(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R, self.gamma, L, D, d)
        print('gamma = {}'.format(self.gamma))
        print('M_d = {}'.format(self.free_vars.M_d))
        print('R = {}'.format(self.free_vars.R))
        print('R @ R.T = {}'.format(self.free_vars.RRt))
        print('jax_result = {}'.format(jax_result))
        print('hardcoded_result = {}'.format(hardcoded_result))
        print('old_result = {}'.format(old_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())

        # Randomized mu and M_d should yield incorrect answers, as jax
        # calculates gradient using M_d while hardcoded uses mu
        self.free_vars.mu = self.rand.random_sample((L, D, 2 * d+1))
        #self.free_vars.M_d = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_R(self.free_vars.mu, self.free_vars.R)[0]
        hardcoded_result = grad_penalty_R(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R, self.gamma, L, D, d)
        self.assertFalse(np.isclose(jax_result, hardcoded_result).all())

class TestGradient(unittest.TestCase):
    def setUp(self):
        L = 6
        D = 2
        d = 4
        gamma = 10

        self.L = L
        self.D = D
        self.d = d
        self.gamma = gamma
        self.poly = ExampleG(D)

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        #mu_vector[0] += 0.001
        mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = np.zeros(M.shape)
        RRt = np.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

    def test_factorization_uniform(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        # TODO change this back 
        gamma = 0

        # change factorization to get nonzero gradient
        self.free_vars.R = np.ones((L, D, d+1, d+1))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        # NOTE I have edited my copy of his code to use d+1 instead
        old_lm = []
        old_lm.append(np.ones((D, L, d+1, d+1)))
        old_lm.append(np.ones((D, L)))
        old_lm.append(np.ones((D, L, d+1)))

        old_mu = np.transpose(np.copy(self.free_vars.mu), axes=(1, 0, 2))
        old_R = np.transpose(np.copy(self.free_vars.R), axes=(1, 0, 2, 3))
        old_x = np.concatenate((old_mu.flatten(), old_R.flatten()))

        #print('old_mu\n{}'.format(old_mu))
        #print('old_R\n{}'.format(old_R))

        new_x = np.concatenate((self.free_vars.mu.flatten(), self.free_vars.R.flatten()))

        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L,
                                         orders_list=powers,
                                         coefficients_list=coef,
                                         Lagrangian_coefficient=old_lm,
                                         rho=gamma)

        old_gradient = jax.grad(aug_lagrangian_partial)

        # Reshape gradient result to be comparable to new gradient
        old_value = old_gradient(old_x)
        old_mu_grad, old_R_grad = restore_matrices(old_value, d, D, L)
        old_mu_grad = np.transpose(old_mu_grad, axes=(1, 0, 2))
        old_R_grad = np.transpose(old_R_grad, axes=(1, 0, 2, 3))

        old_value = np.concatenate((old_mu_grad.flatten(), old_R_grad.flatten()))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('old_mu_grad\n{}'.format(old_mu_grad))
        print('old_R_grad\n{}'.format(old_R_grad))
        print('new_mu_grad\n{}'.format(new_mu_grad))
        print('new_R_grad\n{}'.format(new_R_grad))

        diff = old_value - new_value
        diff_mu = np.copy(diff[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        diff_R = np.copy(diff[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('diff_mu\n{}'.format(diff_mu))
        print('diff_R\n{}'.format(diff_R))

        norm = np.linalg.norm(diff, ord=1)
        print('difference l_1-norm = {}'.format(diff))
        self.assertAlmostEqual(norm, 0, places=2)

    def test_nonnegativity_uniform(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.zeros((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.zeros((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.zeros((D, L, d+1, d+1)))
        old_lm.append(np.ones((D, L)))
        old_lm.append(np.zeros((D, L, d+1)))

        old_mu = np.transpose(np.copy(self.free_vars.mu), axes=(1, 0, 2))
        old_R = np.transpose(np.copy(self.free_vars.R), axes=(1, 0, 2, 3))
        old_x = np.concatenate((old_mu.flatten(), old_R.flatten()))

        new_x = np.concatenate((self.free_vars.mu.flatten(), self.free_vars.R.flatten()))

        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L,
                                         orders_list=powers,
                                         coefficients_list=coef,
                                         Lagrangian_coefficient=old_lm,
                                         rho=gamma)

        old_gradient = jax.grad(aug_lagrangian_partial)

        # Reshape gradient result to be comparable to new gradient
        old_value = old_gradient(old_x)
        old_mu_grad, old_R_grad = restore_matrices(old_value, d, D, L)
        old_mu_grad = np.transpose(old_mu_grad, axes=(1, 0, 2))
        old_R_grad = np.transpose(old_R_grad, axes=(1, 0, 2, 3))


        old_mu_grad = old_mu_grad.flatten()
        old_R_grad = old_R_grad.flatten()
        old_value = np.concatenate((old_mu_grad, old_R_grad))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        diff = old_value - new_value
        diff_mu = np.copy(diff[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        diff_R = np.copy(diff[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        norm = np.linalg.norm(diff, ord=1)
        self.assertAlmostEqual(norm, 0, places=2)

    def test_relaxation_uniform(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.zeros((L, D, d+1, d+1))
        self.lm.nonnegativity = np.zeros((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.zeros((D, L, d+1, d+1)))
        old_lm.append(np.zeros((D, L)))
        old_lm.append(np.ones((D, L, d+1)))

        old_mu = np.transpose(np.copy(self.free_vars.mu), axes=(1, 0, 2))
        old_R = np.transpose(np.copy(self.free_vars.R), axes=(1, 0, 2, 3))
        old_x = np.concatenate((old_mu.flatten(), old_R.flatten()))

        new_x = np.concatenate((self.free_vars.mu.flatten(), self.free_vars.R.flatten()))

        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L,
                                         orders_list=powers,
                                         coefficients_list=coef,
                                         Lagrangian_coefficient=old_lm,
                                         rho=gamma)

        old_gradient = jax.grad(aug_lagrangian_partial)

        # Reshape gradient result to be comparable to new gradient
        old_value = old_gradient(old_x)
        old_mu_grad, old_R_grad = restore_matrices(old_value, d, D, L)
        old_mu_grad = np.transpose(old_mu_grad, axes=(1, 0, 2))
        old_R_grad = np.transpose(old_R_grad, axes=(1, 0, 2, 3))


        old_mu_grad = old_mu_grad.flatten()
        old_R_grad = old_R_grad.flatten()
        old_value = np.concatenate((old_mu_grad, old_R_grad))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        diff = old_value - new_value
        diff_mu = np.copy(diff[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        diff_R = np.copy(diff[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        norm = np.linalg.norm(diff, ord=1)
        self.assertAlmostEqual(norm, 0, places=2)

    def test_zeros(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.ones((D, L, d+1, d+1)))
        old_lm.append(np.ones((D, L)))
        old_lm.append(np.ones((D, L, d+1)))

        old_mu = np.zeros((D, L, 2 * d + 1))
        old_R = np.zeros((D, L, d+1, d+1))
        old_x = np.concatenate((old_mu.flatten(), old_R.flatten()))

        new_x = np.zeros_like(old_x)

        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L,
                                         orders_list=powers,
                                         coefficients_list=coef,
                                         Lagrangian_coefficient=old_lm,
                                         rho=gamma)

        old_gradient = jax.grad(aug_lagrangian_partial)

        # Reshape gradient result to be comparable to new gradient
        old_value = old_gradient(old_x)
        old_mu_grad, old_R_grad = restore_matrices(old_value, d, D, L)
        old_mu_grad = np.transpose(old_mu_grad, axes=(1, 0, 2))
        old_R_grad = np.transpose(old_R_grad, axes=(1, 0, 2, 3))

        old_value = np.concatenate((old_mu_grad.flatten(), old_R_grad.flatten()))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('new_mu_grad\n{}'.format(new_mu_grad))
        print('new_R_grad\n{}'.format(new_R_grad))

        print('old_mu_grad\n{}'.format(old_mu_grad))
        print('old_R_grad\n{}'.format(old_R_grad))

        #R_index = L*D*(2*d+1)
        #print('old[L x D x (2d+1):+6] = {}'.format(old_value[R_index:R_index+6]))
        #print('new[L x D x (2d+1):+6] = {}'.format(new_value[R_index:R_index+6]))
        #print('old_R[:6] = {}'.format(old_R[:6]))

        diff = old_value - new_value
        print('diff = {}'.format(diff))
        norm = np.linalg.norm(diff)
        self.assertAlmostEqual(norm, 0, places=2)

    def test_a(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma
        #gamma = 0

        # we will reshape the data according to our format, as it is stored
        # in Will's (D, L, d) format
        a = np.load('a.npy')
        old_a = np.copy(a)
        new_a = np.copy(a)

        new_mu, new_R = restore_matrices(new_a, d, D, L)
        new_mu = np.transpose(new_mu, axes=(1, 0, 2))
        new_R = np.transpose(new_R, axes=(1, 0, 2, 3))
        new_a = np.concatenate((new_mu.flatten(), new_R.flatten())) #NEW
        self.free_vars.mu = new_mu
        self.free_vars.R = new_R

        print('mu = {}'.format(new_mu))
        print('R = {}'.format(new_R))

        #self.free_vars.mu = np.transpose(new_mu, axes=(1, 0, 2))
        #self.free_vars.R = np.transpose(new_R, axes=(1, 0, 2, 3))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.ones((D, L, d+1, d+1)))
        old_lm.append(np.ones((D, L)))
        old_lm.append(np.ones((D, L, d+1)))

        aug_lagrangian_partial = partial(Augmented_Lagrangian, d=d, D=D, L=L,
                                         orders_list=powers,
                                         coefficients_list=coef,
                                         Lagrangian_coefficient=old_lm,
                                         rho=gamma)

        old_gradient = jax.grad(aug_lagrangian_partial)

        # Reshape gradient result to be comparable to new gradient
        old_value = old_gradient(old_a)
        old_grad_mu, old_grad_R = restore_matrices(old_value, d, D, L)
        old_grad_mu = np.transpose(old_grad_mu, axes=(1, 0, 2))
        old_grad_R = np.transpose(old_grad_R, axes=(1, 0, 2, 3))

        print('old_grad_mu\n{}'.format(old_grad_mu))
        print('old_grad_R\n{}'.format(old_grad_R))

        old_value = np.concatenate((old_grad_mu.flatten(), old_grad_R.flatten()))

        new_value = new_gradient(new_a.flatten(), self.lm, coef, powers, gamma, L, D, d)
        new_grad_mu = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_grad_R = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('new_grad_mu\n{}'.format(new_grad_mu))
        print('new_grad_R\n{}'.format(new_grad_R))
        
        print('mu_diff\n{}'.format(new_grad_mu - old_grad_mu))
        print('R_diff\n{}'.format(new_grad_R - old_grad_R))

        diff = old_value - new_value
        norm = np.linalg.norm(diff, ord=1)
        print('norm diff = {}'.format(norm))
        #self.assertAlmostEqual(norm, 0, places=2)

    def test_d(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # we will reshape the data according to our format, as it is stored
        # in Will's (D, L, d) format
        data = np.load('d.npy')

        # Construct old gradient in new format L, D, d order
        d_grad = np.load('d_grad.npy')
        old_grad_mu, old_grad_R = restore_matrices(d_grad, d, D, L)
        old_grad_mu = np.transpose(old_grad_mu, axes=(1, 0, 2))
        old_grad_R = np.transpose(old_grad_R, axes=(1, 0, 2, 3))

        old_d = np.copy(data)

        new_mu, new_R = restore_matrices(data, d, D, L)
        new_mu = np.transpose(new_mu, axes=(1, 0, 2))
        new_R = np.transpose(new_R, axes=(1, 0, 2, 3))
        new_d = np.concatenate((new_mu.flatten(), new_R.flatten())) #NEW

        self.free_vars.mu = new_mu
        self.free_vars.R = new_R

        print('mu = {}'.format(new_mu))
        print('R = {}'.format(new_R))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        new_value = new_gradient(new_d.flatten(), self.lm, coef, powers, gamma, L, D, d)
        new_grad_mu = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_grad_R = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('old_grad_mu = {}'.format(old_grad_mu))
        print('old_grad_R = {}'.format(old_grad_R))

        print('new_grad_mu = {}'.format(new_grad_mu))
        print('new_grad_R = {}'.format(new_grad_R))

        mu_grad_diff = new_grad_mu - old_grad_mu
        R_grad_diff = new_grad_R - old_grad_R

        mu_diff_norm = np.linalg.norm(mu_grad_diff.flatten(), ord=np.inf)
        R_diff_norm = np.linalg.norm(R_grad_diff.flatten(), ord=np.inf)

        self.assertAlmostEqual(mu_diff_norm, 0, places=3)
        self.assertAlmostEqual(R_diff_norm, 0, places=3)

class TestAugmentedLagrangian(unittest.TestCase):
    def setUp(self):
        L = 6
        D = 2
        d = 4
        gamma = 10

        self.L = L
        self.D = D
        self.d = d
        self.gamma = gamma
        self.poly = ExampleG(D)

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = np.zeros(M.shape)
        RRt = np.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

    #def test_value(self):
    #    coef = self.poly.coefficients
    #    powers = self.poly.powers
    #    #free_vars = self.free_vars.flattened()
    #    new_value = new_augmented_lagrangian(self.x_input, self.lm, coef, powers,
    #                                         self.gamma, self.L, self.D, self.d)
    #    #lagrangian_list = [self.lm.factorization, self.lm.nonnegativity, self.lm.relaxation]
    #    #old_value = Augmented_Lagrangian(self.x_input, self.d, self.D, self.L, powers,
    #    #                                 coef, lagrangian_list, self.gamma)
    #    print(new_value)
    #    #print(old_value)

    def test_a(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        data = np.load('a.npy')

        old_mu, old_R = restore_matrices(data, d, D, L)
        self.free_vars.mu = np.transpose(old_mu, axes=(1, 0, 2))
        self.free_vars.R = np.transpose(old_R, axes=(1, 0, 2, 3))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # from Will's test run
        old_value = jnp.array([1407.7411])
        value = new_augmented_lagrangian(self.free_vars.flattened(), self.lm,
                                         coef, powers, self.gamma, L, D, d)
        # must unpack value from singleton array
        self.assertAlmostEqual(value, old_value, places=3)

    def test_b(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        data = np.load('b.npy')

        old_mu, old_R = restore_matrices(data, d, D, L)
        self.free_vars.mu = np.transpose(old_mu, axes=(1, 0, 2))
        self.free_vars.R = np.transpose(old_R, axes=(1, 0, 2, 3))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # from Will's test run
        old_value = jnp.array(1229.2025)
        value = new_augmented_lagrangian(self.free_vars.flattened(), self.lm,
                                         coef, powers, self.gamma, L, D, d)
        # must unpack value from singleton array
        self.assertAlmostEqual(value, old_value, places=3)

    def test_c(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        data = np.load('c.npy')

        old_mu, old_R = restore_matrices(data, d, D, L)
        self.free_vars.mu = np.transpose(old_mu, axes=(1, 0, 2))
        self.free_vars.R = np.transpose(old_R, axes=(1, 0, 2, 3))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # from Will's test run
        old_value = jnp.array(1304.2158)
        value = new_augmented_lagrangian(self.free_vars.flattened(), self.lm,
                                         coef, powers, self.gamma, L, D, d)
        # must unpack value from singleton array
        self.assertAlmostEqual(value, old_value, places=3)


class TestSolver(unittest.TestCase):
    def setUp(self):
        L = 6
        D = 2
        d = 4
        gamma = 10

        self.L = L
        self.D = D
        self.d = d
        self.gamma = gamma

        # Set reproducible pool of randomness
        self.rand = np.random.RandomState(109332085)

        # Construct moment vector and matrices for uniform distribution over [-1,1]
        mu_vector = np.array([1 / (i+1) if i % 2 == 0 else 0 for i in range(2*d+1)])
        mu = np.array([[np.copy(mu_vector) for d in range(D)] for l in range(L)])
        M = np.array([[[[mu[l,i,n+m] for n in range(d+1)]
                        for m in range(d+1)]
                        for i in range(D)]
                        for l in range(L)])

        R = np.zeros(M.shape)
        RRt = np.zeros(M.shape)

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

    def test_a(self):
        L = 6
        D = 2
        d = 4

        coefficients = (4, 4, -4, -4, 1, 1, 2)
        powers = ((4, 0),
                  (0, 4),
                  (2, 0),
                  (0, 2),
                  (1, 0),
                  (0, 1),
                  (0, 0))
        poly = PolySupport(coefficients, powers)
        solver(poly, self.gamma, L, D, d)
        return

    def paper_example_1(self):
        D = 2
        #gamma = 100_000
        gamma = 1_000
        poly = ExampleF(D)
        print(poly.coefficients)
        print(poly.powers)
        solver(poly, max_iter=5, gamma=gamma)
        return

    def paper_example_2(self):
        D = 4
        gamma = 1_000
        poly = ExampleG(D)
        minimizer = solver(poly, gamma=gamma, max_iter=5, verbose=True)
        truth = -0.75553 * np.ones(4)
        diff = np.linalg.norm(minimizer - truth, ord=1)
        self.assertAlmostEqual(diff, 0, places=2)
        self.assertAlmostEqual(poly.evaluate(minimizer), -1.3911, places=3)

    def experiment_1(self):
        coefficients = (16, 16, -16, -16, 1)
        powers = ((2, 0),
                  (0, 2),
                  (1, 0),
                  (0, 1),
                  (0, 0))
        poly = PolySupport(coefficients, powers)
        solver(poly, gamma=1_000)

    def experiment_2(self):
        coefficients = (16, 16, -16, -16, 1)
        powers = ((4, 0),
                  (0, 4),
                  (2, 0),
                  (0, 2),
                  (0, 0))
        poly = PolySupport(coefficients, powers)
        solver(poly, gamma=1_000)

    def experiment_3(self):
        coefficients = (1, -2, 0)
        powers = ((4,),
                  (2,),
                  (0,))
        poly = PolySupport(coefficients, powers)
        solver(poly, L=2, gamma=1_000)

    def experiment_4(self):
        """
        p(x) = (x^2 - 1/2)^2 + 1
        minimum should be 1
        minimizers are x = +- (1 / sqrt(2))
        """
        coefficients = (1, -1, 1.25)
        powers = ((4,),
                  (2,),
                  (0,))
        poly = PolySupport(coefficients, powers)
        solver(poly, L=2, gamma=1_000)

    def experiment_5(self):
        """
        Testing if solver will converge to a unique solution when the polynomial
        HAS two distinct minimizers BUT convex combinations of their delta distributions
        are not product measures
        """
        coefficients = (1, 1, 2, -2, 0.25)
        powers = ((4, 0),
                  (0, 4),
                  (2, 2),
                  (1, 1),
                  (0, 0))
        poly = PolySupport(coefficients, powers)
        solver(poly, gamma=1_000, seed=None)

    def experiment_6(self):
        """
        Testing the solver with an objective with two completely unrelated
        global minimizers
        """
        coefficients = (32, -16, 64, -48, 12, -16, 16, -4, 32, -48, 28, -8, 1)
        powers = ((4, 0),
                  (3, 0),
                  (2, 2),
                  (2, 1),
                  (2, 0),
                  (1, 2),
                  (1, 1),
                  (1, 0),
                  (0, 4),
                  (0, 3),
                  (0, 2),
                  (0, 1),
                  (0, 0))
        poly = PolySupport(coefficients, powers)
        solver(poly, gamma=1_000, seed=None)
    
class TestHessian(unittest.TestCase):
    def test_1(self):
        L = 2
        D = 1
        d = 4

        coefficients = (1, -2)
        powers = ((4,),
                  (2,))
        poly = PolySupport(coefficients, powers)
        mu = np.load('hessian_test_1.npy')
        hessian = Hessian(poly)

        matrix = hessian.matrix(mu[:,:,:d+1])
        self.assertTrue(np.isclose(matrix, np.zeros((L*D*(d+1), L*D*(d+1)))).all())

    def test_2(self):
        L = 6
        D = 4
        d = 4

        poly = ExampleG(D)
        mu = np.load('hessian_test_2.npy')
        hessian = Hessian(poly)

        matrix = hessian.matrix(mu[:,:,:d+1])
        # test some matrix entries of interest in l = 1 block
        # \del^2 mu_{1,4;2,0}
        self.assertTrue(np.isclose(matrix[4,2*(d+1)], 2, rtol=1e-3))
        # \del^2 mu_{1,2;2,0}
        self.assertTrue(np.isclose(matrix[2,2*(d+1)], -2.070815, rtol=1e-3))
        # \del^2 mu_{2,3;3,0}
        self.assertTrue(np.isclose(matrix[(d+1)+3,3*(d+1)], 0.015625, rtol=1e-3))
        # \del^2 mu_{4,2;3,1}
        self.assertTrue(np.isclose(matrix[3*(d+1)+2,2*(d+1)+1], 0.046875, rtol=1e-3))
        # \del^2 mu_{3,0;4,0}
        self.assertTrue(np.isclose(matrix[2*(d+1),3*(d+1)], -0.034, rtol=1e-3))

        # now a few entries that should be zero
        # the diagonal should be all zeros
        diag = np.diagonal(matrix)
        zeros = np.zeros_like(diag)
        self.assertTrue(np.equal(diag, zeros).all())

        # \del^2 mu_{1,3;2,1}
        self.assertEqual(matrix[3,(d+1) + 1], 0)
        # \del^2 mu_{4,2;1,2}
        self.assertEqual(matrix[3*(d+1)+2, 2], 0)
        # \del^2 mu_{1,4;3,4}
        self.assertEqual(matrix[4,2*(d+1)+4], 0)

    def test_2_decomposition(self):
        L = 6
        D = 4
        d = 4

        poly = ExampleG(D)
        mu = np.load('hessian_test_2.npy')
        print(mu[0,:,:])
        hessian = Hessian(poly)

        matrix = hessian.matrix(mu[:,:,:d+1])
        block = matrix[:D*(d+1), :D*(d+1)]
        evalues, evectors = np.linalg.eigh(block) 
        v = evectors[:,0]
        print(np.reshape(v, (D, d+1)))

class TestCritical(unittest.TestCase):
    def experiment_1(self):
        L = 2
        D = 1
        d = 4

        coefficients = (1, -2, 0)
        powers = ((4,),
                  (2,),
                  (0,))
        poly = PolySupport(coefficients, powers)

        # load minimizer mu from previous run of solver
        mu = np.load('critical_experiment_1.npy')
        print(mu)
        
        hessian = Hessian(poly)
        matrix = hessian.matrix(mu[:,:,:d+1])

        # the Hessian should be all zeros because D = 1
        zeros = np.zeros_like(matrix)
        self.assertTrue(np.equal(matrix, zeros).all())

    def experiment_2(self):
        L = 2
        D = 2
        d = 5

        coefficients = (0.1, 0.1, 1, 1, -1/2, -1/2, 1/8)
        powers = ((5,0),
                  (0,5),
                  (4,0),
                  (0,4),
                  (2,0),
                  (0,2),
                  (0,0))
        poly = PolySupport(coefficients, powers)

        x_min = solver(poly, L=L, gamma=1_000, seed=None)
        print(x_min)

    def experiment_3(self):
        L = 2
        D = 2
        d = 3

        coefficients = (1, 1, 1)
        powers = ((1,3),
                  (2,2),
                  (3,1))
        poly = PolySupport(coefficients, powers)

        x_min = solver(poly, L=L, gamma=1_000, seed=None)

class TestFeasible(unittest.TestCase):
    """
    Tests for functions to evaluate feasibility of mu vectors
    """
    def test_1(self):
        t_values = np.linspace(-1.5, 1.5, 31)
        for t in t_values:
             mu = np.ones(9)
             mu[1] = t
             mu[3] = t
             mu[5] = t
             mu[7] = t
             if np.abs(t) <= 1:
                 self.assertTrue(test_feasible(mu))
             else:
                 self.assertFalse(test_feasible(mu))

class TestPlots(unittest.TestCase):
    def setUp(self):
        pass

    def test_1(self):
        dimensions = [1,]
        for D in dimensions:
            poly = PlotPoly(D)


if __name__ == '__main__':
    unittest.main(verbosity=2)
