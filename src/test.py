import jax
import jax.numpy as jnp
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

        pos_slack = jnp.ones((L, D))
        abs_slack = jnp.zeros((L, D, d+1))

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
            old_mu = np.transpose(mu, axes=(1,0,2)).flatten()
            old_R = np.transpose(R, axes=(1,0,2,3)).flatten()

            old_factorization = np.transpose(lm.factorization, axes=(1, 0, 2, 3))
            old_nonnegativity = np.transpose(lm.nonnegativity, axes=(1, 0))
            old_relaxation = np.transpose(lm.relaxation, axes=(1, 0, 2))
            old_lm = [old_factorization, old_nonnegativity, old_relaxation]
            old = self.old_jax_grad(D, L, old_mu, old_R, old_lm)[1].reshape((D, L, d+1, d+1))
            return np.transpose(old, axes=(1, 0, 2, 3))

        self.old_grad_mu = old_grad_mu
        self.old_grad_R = old_grad_R

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
        jax_result = self.grad_R(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R)
        hardcoded_result = grad_R(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if "random" (but fixed) factorization and lagrange multipliers
        # yield the same answer
        self.lm.factorization = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_R(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R)
        hardcoded_result = grad_R(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

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
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())

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
        self.assertTrue(np.isclose(old_result, hardcoded_result).all())
        self.assertTrue(np.isclose(old_result, jax_result).all())
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
        self.free_vars.mu = 1.001 * np.ones((L, D, 2*d + 1))
        self.free_vars.M_d = 1.001 * self.rand.random_sample((L, D, d+1, d+1))
        self.lm.relaxation = self.rand.random_sample((L, D, d+1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if "random" (but fixed) mu, lagrange multipliers yield the same
        # answer
        self.free_vars.mu = 4 * self.rand.random_sample((L, D, 2*d + 1)) - 2
        self.lm.relaxation = self.rand.random_sample((L, D, d+1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
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

        # specifically check that no infeasibility detected in these positions,
        # as any mu >= 0 should be acceptable
        self.assertTrue(np.equal(jax_result[:,0], np.zeros((L,))).all())
        self.assertTrue(np.equal(hardcoded_result[:,0], np.zeros((L,))).all())

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
        L = 6
        D = 2
        d = 4
        gamma = 1000

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

        pos_slack = np.ones((L, D))
        abs_slack = np.zeros((L, D, d+1))

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

        # gradient with respect to moments and moment matrices
        self.jax_grad_mu = jaxgrad(
                partial(new_penalty, gamma=gamma, L=L, D=D, d=d), argnums=(0, 1))
        # define function to extract raw jax autogradient output and process it to
        # account for the relationship between mu and M_d
        def auto_grad_mu(mu, M_d, R):
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

            return np_mu

        self.grad_mu = auto_grad_mu

        # gradient with respect to R
        self.grad_R = jaxgrad(partial(new_penalty, gamma=gamma, L=L, D=D, d=d), argnums=(2,))

    def test_mu(self):
        """
        Test gradient of factorization constraint with respect to mu
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the existing factorization gap M_d - R @ R.T is registered
        jax_result = self.grad_mu(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R)
        hardcoded_result = grad_penalty_mu(self.free_vars.mu,
                                           self.free_vars.M_d,
                                           self.free_vars.R, self.gamma, L, D,
                                           d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if "random" (but fixed) mu, M_d, and R give the same answer
        # Unlike for the gradients of the multiplier term, here there is no
        # interaction between entries of mu and M_d, so the answers will still
        # be the same even if they have no connection
        self.free_vars.mu = 2 * self.rand.random_sample((L, D, 2 * d+1)) - 1
        self.free_vars.M_d = 6 * self.rand.random_sample((L, D, d+1, d+1)) - 3
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_mu(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R)
        hardcoded_result = grad_penalty_mu(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R, self.gamma, L, D, d)
        #print('gamma = {}'.format(self.gamma))
        #print('M_d = {}'.format(self.free_vars.M_d))
        #print('R @ R.T = {}'.format(self.free_vars.R @ self.free_vars.R))
        #print('jax_result = {}'.format(jax_result))
        #print('hardcoded_result = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

    def test_R(self):
        """
        Test gradient of factorization constraint with respect to R
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the existing factorization gap M_d - R @ R.T is registered
        # TODO add wrapper function to fix [0] unpacking for jax gradient
        jax_result = self.grad_mu(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R)[0]
        hardcoded_result = grad_penalty_mu(self.free_vars.mu,
                                           self.free_vars.M_d,
                                           self.free_vars.R, self.gamma, L, D,
                                           d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if random R gives correct answer
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_R(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R)[0]
        hardcoded_result = grad_penalty_R(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R, self.gamma, L, D, d)
        #print('gamma = {}'.format(self.gamma))
        #print('M_d = {}'.format(self.free_vars.M_d))
        #print('R = {}'.format(self.free_vars.R))
        #print('R @ R.T = {}'.format(self.free_vars.R @ self.free_vars.R))
        #print('jax_result = {}'.format(jax_result))
        #print('hardcoded_result = {}'.format(hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Randomized mu and M_d should yield incorrect answers, as jax
        # calculates gradient using M_d while hardcoded uses mu
        self.free_vars.mu = self.rand.random_sample((L, D, 2 * d+1))
        self.free_vars.M_d = self.rand.random_sample((L, D, d+1, d+1))
        self.free_vars.R = self.rand.random_sample((L, D, d+1, d+1))
        jax_result = self.grad_R(self.free_vars.mu, self.free_vars.M_d,
                                 self.free_vars.R)[0]
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

        pos_slack = np.ones((L, D))
        abs_slack = np.zeros((L, D, d+1))

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)

    def test_factorization_uniform(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # change factorization to get nonzero gradient
        self.free_vars.R = np.ones((L, D, d+1, d+1))

        # change lagrange multipliers to 1 to match his scenario
        self.lm.factorization = np.ones((L, D, d+1, d+1))
        self.lm.nonnegativity = np.zeros((L, D))
        self.lm.relaxation = np.zeros((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.ones((D, L, d+1, d+1)))
        old_lm.append(np.zeros((D, L)))
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

        print('old_mu\n{}'.format(old_mu))
        print('old_R\n{}'.format(old_R))

        old_mu_grad = old_mu_grad.flatten()
        old_R_grad = old_R_grad.flatten()
        old_value = np.concatenate((old_mu_grad, old_R_grad))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('new_mu_grad\n{}'.format(new_mu_grad))
        print('new_R_grad\n{}'.format(new_R_grad))

        diff = old_value - new_value
        diff_mu = np.copy(diff[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        diff_R = np.copy(diff[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('diff_mu\n{}'.format(diff_mu))
        print('diff_R\n{}'.format(diff_R))

        norm = np.linalg.norm(diff, ord=1)
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
        self.lm.nonnegativity = 2 * np.ones((L, D))
        self.lm.relaxation = np.ones((L, D, d+1))

        # translate all 1 Lagrange Multipliers to Will's format
        # NOTE I think he has redundant relaxation constraints 2d+1 instead
        # of just the d+1 specified in the paper. Does this cause problems?
        old_lm = []
        old_lm.append(np.ones((D, L, d+1, d+1)))
        old_lm.append(2 * np.ones((D, L)))
        old_lm.append(np.ones((D, L, 2*d+1)))

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

        old_mu = old_mu.flatten()
        old_R = old_R.flatten()
        old_value = np.concatenate((old_mu, old_R))

        new_value = new_gradient(new_x, self.lm, coef, powers, gamma, L, D, d)
        new_mu_grad = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R_grad = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('new_mu_grad\n{}'.format(new_mu_grad))
        print('new_R_grad\n{}'.format(new_R_grad))

        #print('old[:6] = {}'.format(old_value[:6]))
        #print('new[:6] = {}'.format(new_value[:6]))
        #print('old_mu[:6] = {}'.format(old_mu[:6]))

        #first_offset = L * D * (2*d+1)
        #print('old[LxDx(2d+1):+6] = {}'.format(old_value[first_offset:first_offset+6]))
        #print('new[LxDx(2d+1):+6] = {}'.format(new_value[first_offset:first_offset+6]))
        #print('old_mu[first_offset:+6] = {}'.format(old_mu[first_offset:first_offset+6]))


        #R_index = L*D*(2*d+1)
        #print('old[L x D x (2d+1):+6] = {}'.format(old_value[R_index:R_index+6]))
        #print('new[L x D x (2d+1):+6] = {}'.format(new_value[R_index:R_index+6]))
        #print('old_R[:6] = {}'.format(old_R[:6]))

        diff = old_value - new_value
        norm = np.linalg.norm(diff)
        self.assertAlmostEqual(norm, 0, places=2)

    def test_a(self):
        L = self.L
        D = self.D
        d = self.d

        coef = self.poly.coefficients
        powers = self.poly.powers

        gamma = self.gamma

        # we will reshape the data according to our format, as it is stored
        # in Will's (D, L, d) format
        a = np.load('a.npy')
        old_a = np.copy(a)
        new_a = np.copy(a)

        new_mu, new_R = restore_matrices(new_a, d, D, L)
        self.free_vars.mu = np.transpose(new_mu, axes=(1, 0, 2))
        self.free_vars.R = np.transpose(new_R, axes=(1, 0, 2, 3))

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
        old_mu, old_R = restore_matrices(old_value, d, D, L)
        old_mu = np.transpose(old_mu, axes=(1, 0, 2))
        old_R = np.transpose(new_R, axes=(1, 0, 2, 3))

        print('old_mu\n{}'.format(old_mu))
        print('old_R\n{}'.format(old_R))

        old_value = np.concatenate((old_mu.flatten(), old_R.flatten()))

        new_value = new_gradient(new_a.flatten(), self.lm, coef, powers, gamma, L, D, d)
        new_mu = np.copy(new_value[:L*D*(2*d+1)]).reshape((L, D, 2*d + 1))
        new_R = np.copy(new_value[L*D*(2*d+1):]).reshape((L, D, d+1, d+1))

        print('new_mu\n{}'.format(new_mu))
        print('new_R\n{}'.format(new_R))

        print('mu_diff\n{}'.format(new_mu - old_mu))
        print('R_diff\n{}'.format(new_R - old_R))

        diff = old_value - new_value
        norm = np.linalg.norm(diff, ord=1)
        self.assertAlmostEqual(norm, 0, places=2)


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

        pos_slack = np.ones((L, D))
        abs_slack = np.zeros((L, D, d+1))

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

        pos_slack = np.ones((L, D))
        abs_slack = np.zeros((L, D, d+1))

        self.free_vars = FreeVariables(L, D, d, mu, R)

        # lambda
        self.lm = LagrangeMultipliers(L, D, d)


    def test_example_2(self):
        poly = ExampleG(self.D)
        solver(poly, self.gamma, self.L, self.D, self.d)
        return

    

if __name__ == '__main__':
    unittest.main(verbosity=2)
