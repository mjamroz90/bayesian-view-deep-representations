import numpy as np


def multivariate_t_rvs(m, s_chol, df):
    m = np.asarray(m)
    d = len(m)
    if df == np.inf:
        x = 1.
    else:
        x = np.random.chisquare(df)/df

    std_normal = np.random.randn(d)
    z = np.dot(s_chol, std_normal)
    return m + z/np.sqrt(x)


def multivariate_t_rvs_full_and_cov(m, s_chol, df, samples_num):
    m = np.asarray(m)
    x = np.random.chisquare(df, size=samples_num)/df

    std_normal = np.random.randn(m.shape[0], samples_num)
    s_chol_diag = np.diag(np.diag(s_chol))

    z = np.dot(s_chol, std_normal)
    z_diag = np.dot(s_chol_diag, std_normal)
    # z.shape = (d, samples_num)
    x = np.expand_dims(x, axis=0)
    m = np.expand_dims(m, axis=-1)
    result = {'sample_full': (m + z/np.sqrt(x)).T, 'sample_diag': (m + z_diag/np.sqrt(x)).T}

    return result


def cov_error_ellipse(mean, cov, p, samples_num=100):
    assert mean.shape[0] == 2

    s = -2. * np.log(1. - p)
    vals, vecs = np.linalg.eig(cov * s)

    t = np.linspace(0, 2 * np.pi, num=samples_num)
    rs = vecs * np.expand_dims(np.sqrt(vals), axis=0)

    t_vals = np.dot(rs, [np.cos(t), np.sin(t)])

    xs = mean[0] + t_vals[0, :]
    ys = mean[1] + t_vals[1, :]

    return xs, ys
