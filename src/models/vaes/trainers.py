import os.path as op
import tensorflow as tf

import numpy as np

from utils.logger import log
from utils import tf_ops


@log
class VaeTrainer(object):
    def __init__(self, dataset, out_weights_dir, vae_model, **kwargs):
        self.out_weights_dir = out_weights_dir
        self.batch_size = dataset.batch_size
        self.beta = kwargs['beta']

        self.img_size = dataset.img_size()
        self.reg_type = kwargs['reg_type'] if 'reg_type' in kwargs else 'kl'
        self.z_sigma_sq = kwargs['z_sigma_sq'] if 'z_sigma_sq' in kwargs else 1.

        assert self.reg_type in {'kl', 'mmd-sq', 'mmd-imq'}

        if self.reg_type == 'kl' and self.z_sigma_sq != 1.:
            raise ValueError("z_sigma_sq can be different than 1. only if regularization is MMD*")

        self.kwargs = kwargs
        self.vae_model = vae_model

        if self.kwargs['ds']['scale_img']:
            self.transform_func = lambda x: tf.nn.sigmoid(x)
        else:
            self.transform_func = lambda x: x

        self.weights_dump_interval = 2000
        self.dataset = dataset

    @tf.function
    def losses(self, input_images, transform_func, reg_type):

        enc_out = self.vae_model.encode_from_input(input_images)
        z = self.vae_model.reparameterize(enc_out)
        dec_out = self.vae_model.decode_from_z(z)
        recon_loss = self.vae_model.recon_loss(input_images, dec_out, transform_func)

        if reg_type == 'kl':
            reg_loss = self.vae_model.kl_loss(enc_out)
        elif reg_type.startswith('mmd'):
            stddev = np.sqrt(self.z_sigma_sq)
            std_normal_samples = tf.random.normal(shape=tf.shape(z), mean=0.0, stddev=stddev,
                                                  dtype=tf.float32)
            if reg_type == 'mmd-sq':
                reg_loss = tf_ops.compute_sq_exp_mmd(z, std_normal_samples)
            else:
                reg_loss = tf_ops.compute_imq_mmd(z, std_normal_samples)
        else:
            raise ValueError("Unknown reg_type == %s" % self.reg_type)

        vae_loss = recon_loss + self.beta * reg_loss
        return reg_loss, recon_loss, vae_loss

    @tf.function
    def train_step(self, input_images, optimizer):
        with tf.GradientTape() as tape:
            reg_loss, recon_loss, vae_loss = self.losses(input_images, self.transform_func, self.reg_type)

        trainable_variables = self.vae_model.model_variables()
        gradients = tape.gradient(vae_loss, trainable_variables)
        optimizer.apply_gradients(zip(gradients, trainable_variables))

        return reg_loss, recon_loss, vae_loss

    def save_enc_dec_models(self, out_weights_dir, iter_num):
        out_path = op.join(out_weights_dir, "model-%d.%s" % (iter_num, self.vae_model.out_file_ext()))
        self.vae_model.save_params_to_file(out_path)

    def train(self, epochs_num):
        if self.kwargs['gc'] is not None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-4, clipvalue=self.kwargs['gc'])
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-4)

        ds_size = self.dataset.train_ds_size()
        batches_num = ds_size // self.batch_size

        if 'restore_model_path' in self.kwargs:
            iter_counter = int(op.splitext(op.basename(self.kwargs['restore_model_path']))[0].split('-')[-1])
            self.vae_model.load_params(self.kwargs['restore_model_path'], False)
            self.logger.info("Restoring training state from path: %s, global-step: %d" %
                             (self.kwargs['restore_model_path'], iter_counter))
        else:
            iter_counter = 0
            self.logger.info("Initializing everything from scratch")

        curr_epoch = int(iter_counter / batches_num)

        for epoch in range(curr_epoch, epochs_num, 1):
            for batch_counter, train_examples in enumerate(self.dataset.generate_train_mb()):
                self.logger.info("Started %d/%d epoch" % (epoch, epochs_num))
                reg_loss, recon_loss, vae_loss = self.train_step(train_examples, optimizer)

                self.logger.info("Epoch: %d/%d, batch: %d/%d, VAE loss: %.5f, Recon loss: %.5f, Reg(%s) loss: %.5f" %
                                 (epoch, epochs_num, batch_counter, batches_num, vae_loss.numpy(),
                                  recon_loss.numpy(), self.reg_type,
                                  reg_loss.numpy()))

                if iter_counter % self.weights_dump_interval == 0:
                    self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
                    self.save_enc_dec_models(self.out_weights_dir, iter_counter)

                iter_counter += 1

            self.logger.info("Finished epoch %d" % epoch)
            test_loss = self.test_model()

            self.logger.info("After testing model from iteration: %d - loss: %.5f" % (iter_counter, test_loss))
            self.logger.info("Iteration %d, dumping parameters ..." % iter_counter)
            self.save_enc_dec_models(self.out_weights_dir, iter_counter)

    def test_model(self):
        test_loss = 0.
        self.logger.info("Testing model ...")
        iterations_num = 0
        for batch_counter, test_examples in enumerate(self.dataset.generate_test_mb()):
            batch_reg, batch_recon, batch_loss = self.losses(test_examples, self.transform_func, self.reg_type)

            iterations_num += 1
            test_loss += batch_loss

            self.logger.info("Testing model, iteration: %d, loss: %.5f, recon: %.5f, reg(%s) loss: %.5f" %
                             (batch_counter, batch_loss.numpy(), batch_recon.numpy(), self.reg_type, batch_reg.numpy()))

        return float(test_loss) / iterations_num

