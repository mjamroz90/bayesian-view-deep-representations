import tensorflow as tf


def sq_exp(x, y, sigma_sq):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    euclidean_dist = tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)
    euclidean_dist = tf.math.minimum(euclidean_dist, 1.e6)

    return tf.exp(-euclidean_dist / (2.*sigma_sq))


def inverse_multi_quadratic(x, y, c_const):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))

    euclidean_dist = tf.reduce_sum(tf.square(tiled_x - tiled_y), axis=2)
    euclidean_dist = tf.math.minimum(euclidean_dist, 1.e6)
    return c_const / (euclidean_dist + c_const)


def compute_mmd(x, y, kernel_func):
    x_kernel = kernel_func(x, x)
    y_kernel = kernel_func(y, y)
    xy_kernel = kernel_func(x, y)
    return tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2. * tf.reduce_mean(xy_kernel)


def compute_sq_exp_mmd(x, y, sigma_sq=2.):
    dim = tf.cast(tf.shape(x)[1], tf.float32)
    return compute_mmd(x, y, lambda t1, t2: sq_exp(t1, t2, sigma_sq=2. * dim * sigma_sq))


def compute_imq_mmd(x, y, sigma_sq=2.):
    dim = tf.cast(tf.shape(x)[1], tf.float32)
    return compute_mmd(x, y, lambda v1, v2: inverse_multi_quadratic(v1, v2, 2. * dim * sigma_sq))

