import os, math
import tensorflow as tf
import numpy as np
import gc
from tensorflow.keras import losses, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import clone_model

import utils
import mapping, generator, discriminator


class GAN(Model):

	def __init__(self, latent_dim, image_size, nb_channels, mapping_layers, mapping_lr_ratio, min_image_size, max_filters, min_filters,
		kernel_size, alpha, gain, style_mix_proba, gp_coef, gp_interval, ma_beta, nb_batchs, **kwargs):

		super(GAN, self).__init__(**kwargs)

		self.mapping = mapping.build_mapping(latent_dim, mapping_layers, mapping_lr_ratio, alpha, gain)
		self.generator = generator.build_generator(latent_dim, image_size, nb_channels, min_image_size, max_filters, min_filters, kernel_size, alpha, gain)
		self.discriminator = discriminator.build_discriminator(image_size, nb_channels, min_image_size, max_filters, min_filters, kernel_size, alpha, gain)

		self.ma_mapping = clone_model(self.mapping)
		self.ma_mapping.set_weights(self.mapping.get_weights())

		self.ma_generator = clone_model(self.generator)
		self.ma_generator.set_weights(self.generator.get_weights())

		self.latent_dim = latent_dim
		self.image_size = image_size
		self.nb_channels = nb_channels
		self.style_mix_proba = style_mix_proba
		self.gp_coef = gp_coef
		self.gp_interval = gp_interval
		self.ma_beta = ma_beta
		self.nb_batchs = nb_batchs

		self.batch = tf.Variable(0, dtype = tf.int32, trainable = False, name = "batch")
		self.epoch = tf.Variable(0, dtype = tf.int32, trainable = False, name = "epoch")
		self.nb_blocks = int(math.log(image_size, 2)) - int(math.log(min_image_size, 2)) + 1


	def compile(self, learning_rate, beta_1, beta_2, epsilon, **kwargs):

		super(GAN, self).compile(**kwargs)
		self.generator_optimizer = Adam(learning_rate, beta_1, beta_2, epsilon)
		self.discriminator_optimizer = Adam(learning_rate, beta_1, beta_2, epsilon)


	def save_weights(self, dir, i):

		path = os.path.join(dir, "model_" + str(i))

		if not os.path.exists(path):
			os.makedirs(path)

		self.mapping.save_weights(os.path.join(path, "mapping.h5"))
		self.generator.save_weights(os.path.join(path, "generator.h5"))
		self.discriminator.save_weights(os.path.join(path, "discriminator.h5"))
		self.ma_mapping.save_weights(os.path.join(path, "ma_mapping.h5"))
		self.ma_generator.save_weights(os.path.join(path, "ma_generator.h5"))


	def load_weights(self, dir):

		folder = ""
		i = 1

		while True:

			if os.path.exists(os.path.join(dir, "model_" + str(i))):
				folder = os.path.join(dir, "model_" + str(i))

			else:
				break

			i += 1

		if folder != "":
			self.mapping.load_weights(os.path.join(folder, "mapping.h5"))
			self.generator.load_weights(os.path.join(folder, "generator.h5"))
			self.discriminator.load_weights(os.path.join(folder, "discriminator.h5"))
			self.ma_mapping.load_weights(os.path.join(folder, "ma_mapping.h5"))
			self.ma_generator.load_weights(os.path.join(folder, "ma_generator.h5"))

		return i - 1


	def moving_average(self):

		for i in range(len(self.mapping.layers)):

			weights = self.mapping.layers[i].get_weights()
			old_weights = self.ma_mapping.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * self.ma_beta + (1. - self.ma_beta) * weights[j])

			self.ma_mapping.layers[i].set_weights(new_weights)

		for i in range(len(self.generator.layers)):

			weights = self.generator.layers[i].get_weights()
			old_weights = self.ma_generator.layers[i].get_weights()
			new_weights = []

			for j in range(len(weights)):
				new_weights.append(old_weights[j] * self.ma_beta + (1. - self.ma_beta) * weights[j])

			self.ma_generator.layers[i].set_weights(new_weights)


	def predict(self, z, noise, batch_size):

		generations = np.zeros((z.shape[0], self.image_size, self.image_size, self.nb_channels), dtype = np.uint8)

		for i in range(0, z.shape[0], batch_size):

			size = min(batch_size, z.shape[0] - i)
			const_input = [tf.ones((size, 1))]
			w = tf.convert_to_tensor(self.ma_mapping(z[i:i + size]))
			n = [tf.convert_to_tensor(j[i:i + size]) for j in noise]
			gen = self.ma_generator(const_input + ([w] * self.nb_blocks) + n)
			generations[i:i + size, :, :, :] = utils.denorm_img(gen.numpy())

		return generations


	def get_w(self, batch_size):

		rand = tf.random.uniform(shape = (), minval = 0., maxval = 1., dtype = tf.float32)

		if rand < self.style_mix_proba:

			cross_over_point = tf.random.uniform(shape = (), minval = 1, maxval = self.nb_blocks, dtype = tf.int32)

			z1 = tf.random.normal(shape = (batch_size, self.latent_dim))
			z2 = tf.random.normal(shape = (batch_size, self.latent_dim))

			w1 = self.mapping(z1, training = True)
			w2 = self.mapping(z2, training = True)
			w = []

			for i in range(self.nb_blocks):

				if i < cross_over_point:
					w_i = w1

				else:
					w_i = w2

				w.append(w_i)

			return w

		else:

			z = tf.random.normal(shape = (batch_size, self.latent_dim))
			w = self.mapping(z, training = True)

			return [w] * self.nb_blocks


	def get_noise(self, batch_size):

		return [tf.random.normal((batch_size, self.image_size, self.image_size, 1)) for _ in range((self.nb_blocks * 2) - 1)]


	def generator_loss(self, fake_output):

		return tf.reduce_mean(tf.nn.softplus(-fake_output))


	def discriminator_loss(self, real_output, fake_output):

		return tf.reduce_mean(tf.nn.softplus(-real_output)) + tf.reduce_mean(tf.nn.softplus(fake_output))


	def gradient_penalty(self, real_output, real_images):

		gradients = tf.gradients(real_output, real_images)[0]
		gradient_penalty = tf.reduce_sum(tf.square(gradients), axis = tf.range(1, len(gradients.shape)))

		return tf.reduce_mean(gradient_penalty) * self.gp_coef * 0.5 * self.gp_interval


	@tf.function
	def train_step(self, data):

		batch_size = tf.shape(data)[0]
		const_input = [tf.ones((batch_size, 1))]
		noise = self.get_noise(batch_size)

		with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

			w = self.get_w(batch_size)
			fake_images = self.generator(const_input + w + noise, training = True)
			real_output = self.discriminator(data, training = True)
			fake_output = self.discriminator(fake_images, training = True)

			gen_loss = self.generator_loss(fake_output)
			disc_loss = self.discriminator_loss(real_output, fake_output)

			if (self.epoch * self.nb_batchs + self.batch) % self.gp_interval == 0:
				gradient_penalty = self.gradient_penalty(real_output, data)
			else:
				gradient_penalty = 0.

			disc_loss += gradient_penalty

			trainable_weights = (self.mapping.trainable_weights + self.generator.trainable_weights)

			generator_grad = gen_tape.gradient(gen_loss, trainable_weights)
			discriminator_grad = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

			self.generator_optimizer.apply_gradients(zip(generator_grad, trainable_weights))
			self.discriminator_optimizer.apply_gradients(zip(discriminator_grad, self.discriminator.trainable_variables))

		return {
			"Generator loss": gen_loss,
			"Discriminator loss": disc_loss,
			"Gradient penalty": gradient_penalty
		}
