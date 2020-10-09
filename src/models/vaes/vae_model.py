import abc

import tensorflow as tf

from src.models.vaes import delta_vae
from utils import fs_utils


class VaeModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass

    @abc.abstractmethod
    def encode_from_input(self, input_images):
        return

    @abc.abstractmethod
    def decode_from_z(self, z_samples):
        return

    @abc.abstractmethod
    def recon_loss(self, input_images, decoder_out, transform_func):
        return

    @abc.abstractmethod
    def save_params_to_file(self, out_file_path):
        return

    @abc.abstractmethod
    def load_params(self, model_file_path, only_dec):
        return

    @abc.abstractmethod
    def model_variables(self):
        return

    @abc.abstractmethod
    def out_file_ext(self):
        return

    def kl_loss(self, enc_out):
        enc_info = self.split_enc_out(enc_out)
        kl_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + enc_info['cov'] - tf.square(enc_info['mean'])
                                                      - tf.exp(enc_info['cov']), axis=1))
        return kl_loss

    def reparameterize(self, enc_out):
        enc_info = self.split_enc_out(enc_out)
        eps = tf.random.normal(shape=(tf.shape(enc_out)[0], enc_info['latent_dim']), mean=0.0, stddev=1.0)
        z = enc_info['mean'] + tf.multiply(tf.sqrt(tf.exp(enc_info['cov'])), eps)

        return z

    def split_enc_out(self, enc_out):
        latent_dim = int(enc_out.get_shape()[-1] / 2)
        return {'mean': enc_out[:, :latent_dim], 'cov': enc_out[:, latent_dim:], 'latent_dim': latent_dim}


class StandardVae(VaeModel):

    def __init__(self, input_shape, latent_dim, arch_func, delta_val=None, dec_sigma_sq=1.):
        super().__init__()

        self.input_shape = input_shape
        self.latent_dim = latent_dim
        self.delta_val = delta_val
        self.dec_sigma_sq = dec_sigma_sq

        self.enc_model, self.dec_model = arch_func()

    def save_params_to_file(self, out_file_path):
        out_enc_path = fs_utils.add_suffix_to_path(out_file_path, "enc")
        out_dec_path = fs_utils.add_suffix_to_path(out_file_path, 'dec')

        self.enc_model.save_weights(out_enc_path)
        self.dec_model.save_weights(out_dec_path)

    def load_params(self, model_file_path, only_dec):
        enc_model_path = fs_utils.add_suffix_to_path(model_file_path, "enc")
        dec_model_path = fs_utils.add_suffix_to_path(model_file_path, "dec")

        self.dec_model.load_weights(dec_model_path)
        if not only_dec:
            self.enc_model.load_weights(enc_model_path)

    def model_variables(self):
        return self.enc_model.trainable_variables + self.dec_model.trainable_variables

    @delta_vae.keepabovedelta
    def encode_from_input(self, input_images_ph):
        return self.enc_model(input_images_ph)

    def decode_out(self, input_images):
        enc_out = self.encode_from_input(input_images)
        z = self.reparameterize(enc_out)
        return self.decode_from_z(z)

    def decode_from_z(self, z_samples):
        return self.dec_model(z_samples)

    def recon_loss(self, input_images, decoder_out, transform_func=lambda x: x):
        factor = tf.constant(1., dtype=tf.float32) if self.dec_sigma_sq == 1 \
            else tf.constant(1./(2. * self.dec_sigma_sq), dtype=tf.float32)
        pixels_euclidean_dists = tf.square(input_images - transform_func(decoder_out))
        pixels_euclidean_dists = tf.minimum(pixels_euclidean_dists, 1.e4)
        return tf.reduce_mean(tf.math.reduce_sum(pixels_euclidean_dists, axis=[1, 2, 3]))*factor

    def out_file_ext(self):
        return "h5"
