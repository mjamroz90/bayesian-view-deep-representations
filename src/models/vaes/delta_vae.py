import numpy as np
from scipy import optimize
import tensorflow as tf

from utils.logger import log


class DeltaVae(object):

    def __init__(self, delta_val):
        self.delta_val = delta_val

        assert delta_val > 0.

        self.left_int, self.right_int = self.find_cov_intervals()

    def reparameterize_to_save_constraint(self, mean, log_cov):
        # Reparameterization is as follows:
        # new_sigma_q2 = sigma_q2_l + (sigma_q2_u - sigma_q2_l)*(1 / (1 + exp(-sigma_q2(x))))
        # new_mu_q = 2 * delta_val + 1 + ln(new_sigma_q2) - new_sigma_q2 + max(0, mu_q(x))
        # cov variable is in real a logarithm of covariance
        cov = tf.math.exp(log_cov)
        cov_bounded_0_to_1 = tf.math.sigmoid(cov)

        new_sigma_cov = self.left_int + (self.right_int - self.left_int)*cov_bounded_0_to_1
        new_sigma_log_cov = tf.math.log(new_sigma_cov)

        new_mean = 2. * self.delta_val + 1. + new_sigma_log_cov - new_sigma_cov + tf.maximum(0., mean)

        return new_mean, new_sigma_log_cov

    def find_cov_intervals(self):
        # Condition for D_KL(q || N(0,1)) > delta_val:
        # mu_q^2 > 2*delta_val + 1 + ln(sigma_q2) - sigma_q2
        #
        # First, we have to find the interval for sigma_q: [sigma_q2_l, sigma_q2_u] where the RHS is greater than 0
        # So, we have to find roots of a following non-linear equation:
        # ln(sigma_q2) - sigma_q2 + 2*delta_val + 1 = 0
        # The above function is concave and achieves maximum at sigma_q2 = 1
        # In order to find sigma_q2_l and sigma_q2_u, we have to run Newton-Raphson algorithm twice:
        # - 1. starting at initial point close to 0, to find sigma_q2_l
        # - 2. starting at initial point greater than 1, to ding sigma_q2_u

        def f(x):
            return np.log(x) - x + 2. * self.delta_val + 1.

        def f_prime(x):
            return 1./x - 1.

        def f_prime2(x):
            return -1./(x*x)

        low_end_x0, up_end_x0 = 0.1, 1.5
        left_root = optimize.newton(f, low_end_x0, fprime=f_prime, tol=1.e-4, maxiter=200, fprime2=f_prime2)
        right_root = optimize.newton(f, up_end_x0, fprime=f_prime, tol=1.e-4, maxiter=200, fprime2=f_prime2)

        return left_root, right_root


@log
def keepabovedelta(func):
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'delta_val'):
            raise ValueError("Object of the class %s must have an attribute delta_val" % str(type(self)))

        split_enc_out_func = getattr(self, 'split_enc_out', None)
        if not callable(split_enc_out_func):
            raise ValueError("Object of class ")

        enc_out = func(self, *args, **kwargs)
        if self.delta_val is None:
            keepabovedelta.logger.info("Delta_val is None, nothing to reparameterize")
            return enc_out

        delta_val_obj = DeltaVae(self.delta_val)
        enc_out_split = self.split_enc_out(enc_out)
        new_mean, new_cov = delta_val_obj.reparameterize_to_save_constraint(enc_out_split['mean'],
                                                                            enc_out_split['cov'])

        keepabovedelta.logger.info("Reparameterized mean and log-covariance to keep KL value above %.2f"
                                   % self.delta_val)
        return tf.concat((new_mean, new_cov), axis=-1)

    return wrapper



