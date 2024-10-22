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

        self.lm.factorization = self.lm.factorization.at[0,0,3,1].set(1)
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        # Should add nothing to product, no violation here
        self.lm.factorization = self.lm.factorization.at[0,0,2,1].set(1)
        self.assertEqual(self.lm.multiply(self.free_vars), 1/5)

        self.lm.factorization = self.lm.factorization.at[0,0,4,2].set(-1)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 1/5 - 1/7)

        # Fix violation in M_d[0,0,3,1]
        self.free_vars.R = self.free_vars.R.at[0,0,3,0].set(1 / 5)
        self.free_vars.R = self.free_vars.R.at[0,0,1,0].set(1)
        self.free_vars.update_RRt()
        self.assertEqual(self.lm.multiply(self.free_vars), -1/7)


    def test_nonnegativity_penalty(self):
        self.lm.nonnegativity = self.lm.nonnegativity.at[1,0].set(3.7)
        self.free_vars.mu = self.free_vars.mu.at[1,0,0].set(-1.9)
        self.assertEqual(self.lm.multiply(self.free_vars), -7.03)

    def test_mu_equality_constraints(self):
        self.lm.nonnegativity = self.lm.nonnegativity.at[1,1].set(5.8)
        self.free_vars.mu = self.free_vars.mu.at[1,1,0].set(8.03)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        # should not change value
        self.lm.nonnegativity = self.lm.nonnegativity.at[0,1].set(-1_000)
        self.assertAlmostEqual(self.lm.multiply(self.free_vars), 40.774)

        self.free_vars.mu = self.free_vars.mu.at[0,1,0].set(0.5)
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


    def test(self):
        """
        Test gradient of objective function
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if the existing factorization gap M_d - R @ R.T is registered
        jax_result = self.grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if random mu gives correct answer
        print('random\n========')
        self.mu = 2 * self.rand.random_sample((L, D, 2*d + 1)) - 1
        jax_result = self.grad(self.mu)
        hardcoded_result = grad_objective(self.mu,
                                          self.p.coefficients,
                                          self.p.powers,
                                          L, D, d)
        print('jax = {}'.format(jax_result))
        print('hardcoded = {}'.format(hardcoded_result))
        print('diff = {}'.format(jax_result - hardcoded_result))
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())


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

        pos_slack = np.ones((L, D))
        abs_slack = np.zeros((L, D, d+1))

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

    def test_factorization_R(self):
        """
        Test gradient of factorization constraint with respect to factorization
        matrix R
        """
        L = self.L
        D = self.D
        d = self.d

        # Test if one lagrange multiplier factor and one R element change works
        self.lm.factorization = self.lm.factorization.at[0,1,1,3].set(0.5)
        self.free_vars.R = self.free_vars.R.at[0,1,1,1].set(1)
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
        self.lm.factorization = self.lm.factorization.at[0,1,1,3].set(0.5)
        self.free_vars.R = self.free_vars.R.at[0,1,1,1].set(1)
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
        self.free_vars.mu = self.rand.random_sample((L, D, 2*d + 1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
        self.assertTrue(np.isclose(jax_result, hardcoded_result).all())

        # Test if "random" (but fixed) factorization and lagrange multipliers
        # yield the same answer
        self.lm.nonnegativity = self.rand.random_sample((L, D))
        self.free_vars.mu = 5 * self.rand.random_sample((L, D, 2*d + 1))
        jax_result = self.grad_mu(self.lm.factorization, self.lm.nonnegativity,
                          self.lm.relaxation, self.free_vars.mu,
                          self.free_vars.M_d, self.free_vars.R, L, D, d)
        hardcoded_result = grad_mu(self.lm.factorization, self.lm.nonnegativity,
                                 self.lm.relaxation, self.free_vars.mu,
                                 self.free_vars.R, L, D,
                                 d)
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
        L = 2
        D = 2
        d = 4
        gamma = 2

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

    

if __name__ == '__main__':
    unittest.main(verbosity=2)
