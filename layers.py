import math
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomNormal, Constant


def get_filters(max_filters, min_filters, nb_blocks, max_first):

	filters = []

	for i in range(nb_blocks):
		filters.append(min(int(min_filters * (2 ** i)), max_filters))

	if max_first:
		return filters[::-1]

	return filters


class PixelNorm(Layer):

	def __init__(self, epsilon = 1e-8, **kwargs):

		super(PixelNorm, self).__init__(**kwargs)
		self.epsilon = epsilon


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'epsilon': self.epsilon
		})

		return config


	def call(self, inputs):

		return inputs * tf.math.rsqrt(tf.reduce_mean(tf.square(inputs), axis = -1, keepdims = True) + self.epsilon)


class EqualizedDense(Layer):

	def __init__(self, units, bias_init = 0., use_bias = True, gain = 1., lr_multiplier = 1., **kwargs):

		super(EqualizedDense, self).__init__(**kwargs)
		self.units = units
		self.bias_init = bias_init
		self.use_bias = use_bias
		self.gain = gain
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'units': self.units,
			'bias_init': self.bias_init,
			'use_bias': self.use_bias,
			'gain': self.gain,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super(EqualizedDense, self).build(input_shape)

		self.weight = self.add_weight(
			shape = (input_shape[-1], self.units),
			initializer = RandomNormal(mean = 0., stddev = 1. / self.lr_multiplier),
			trainable = True,
			name = "kernel"
		)

		if self.use_bias:
			self.bias = self.add_weight(
				shape = (self.units,),
				initializer = Constant(self.bias_init / self.lr_multiplier),
				trainable = True,
				name = "bias"
			)

		fan_in = input_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		output = tf.matmul(inputs, self.scale * self.weight)

		if self.use_bias:
			return (output + self.bias) * self.lr_multiplier

		return output * self.lr_multiplier


class EqualizedConv2D(Layer):

	def __init__(self, filters, kernel_size, bias_init = 0., use_bias = True, gain = 1., lr_multiplier = 1., **kwargs):

		super(EqualizedConv2D, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.bias_init = bias_init
		self.use_bias = use_bias
		self.gain = gain
		self.lr_multiplier = lr_multiplier


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'bias_init': self.bias_init,
			'use_bias': self.use_bias,
			'gain': self.gain,
			'lr_multiplier': self.lr_multiplier
		})

		return config


	def build(self, input_shape):

		super(EqualizedConv2D, self).build(input_shape)

		self.kernel = self.add_weight(
			shape = (self.kernel_size, self.kernel_size, input_shape[-1], self.filters),
			initializer = RandomNormal(mean = 0., stddev = 1. / self.lr_multiplier),
			trainable = True,
			name = "kernel",
		)

		if self.use_bias:

			self.bias = self.add_weight(
				shape = (self.filters,),
				initializer = Constant(self.bias_init / self.lr_multiplier),
				trainable = True,
				name = "bias"
			)

		fan_in = self.kernel_size * self.kernel_size * input_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		if self.kernel_size > 1:
			inputs = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode = "REFLECT")

		output = tf.nn.conv2d(inputs, self.scale * self.kernel, strides = 1, padding = "VALID", data_format = "NHWC")

		if self.use_bias:
			return (output + self.bias[None, None, None, :]) * self.lr_multiplier

		return output * self.lr_multiplier


class AddNoise(Layer):

	def __init__(self, **kwargs):

		super(AddNoise, self).__init__(**kwargs)


	def build(self, input_shape):

		super(AddNoise, self).build(input_shape)

		self.noise_scale = self.add_weight(
			shape = None,
			initializer = Constant(0.),
			trainable = True,
			name = "noise_scale",
		)


	def call(self, inputs):

		return inputs[0] + (self.noise_scale * inputs[1])


class Bias(Layer):

	def __init__(self, bias_init = 0., **kwargs):

		super(Bias, self).__init__(**kwargs)
		self.bias_init = bias_init


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'bias_init': self.bias_init
		})

		return config


	def build(self, input_shape):

		super(Bias, self).build(input_shape)

		self.bias = self.add_weight(
			shape = (input_shape[-1],),
			initializer = Constant(self.bias_init),
			trainable = True,
			name = "bias"
		)


	def call(self, inputs):

		return inputs + self.bias[None, None, None, :]


class ModulatedConv2D(Layer):

	def __init__(self, filters, kernel_size, demodulate = True, epsilon = 1e-8, gain = 1., **kwargs):

		super(ModulatedConv2D, self).__init__(**kwargs)
		self.filters = filters
		self.kernel_size = kernel_size
		self.demodulate = demodulate
		self.epsilon = epsilon
		self.gain = gain


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'filters': self.filters,
			'kernel_size': self.kernel_size,
			'demodulate': self.demodulate,
			'epsilon': self.epsilon,
			'gain': self.gain
		})

		return config


	def build(self, input_shape):

		super(ModulatedConv2D, self).build(input_shape)

		x_shape = input_shape[0]

		self.kernel = self.add_weight(
			shape = (self.kernel_size, self.kernel_size, x_shape[-1], self.filters),
			initializer = RandomNormal(mean = 0., stddev = 1.),
			trainable = True,
			name = "kernel",
		)

		fan_in = self.kernel_size * self.kernel_size * x_shape[-1]
		self.scale = self.gain / np.sqrt(fan_in)


	def call(self, inputs):

		x = inputs[0]
		style = inputs[1]

		x = tf.transpose(x, [0, 3, 1, 2])

		# Modulate
		ww = (self.scale * self.kernel)[None, :, :, :, :]
		ww *= style[:, None, None, :, None]

		# Demodulate
		if self.demodulate:
			sigma = tf.math.rsqrt(tf.reduce_sum(tf.square(ww), axis = [1, 2, 3]) + self.epsilon)
			ww *= sigma[:, None, None, None, :]

		# Reshape input
		x = tf.reshape(x, (1, -1, x.shape[2], x.shape[3]))
		w = tf.reshape(tf.transpose(ww, [1, 2, 3, 0, 4]), (ww.shape[1], ww.shape[2], ww.shape[3], -1))

		# Convolution
		output = tf.nn.conv2d(x, w, strides = 1, padding = "SAME", data_format = "NCHW")

		# Reshape output
		output = tf.reshape(output, (-1, self.filters, x.shape[2], x.shape[3]))

		return tf.transpose(output, [0, 2, 3, 1])


class MinibatchStdDev(Layer):

	def __init__(self, epsilon = 1e-8, **kwargs):

		super(MinibatchStdDev, self).__init__(**kwargs)
		self.epsilon = epsilon


	def get_config(self):

		config = super().get_config().copy()

		config.update({
			'epsilon': self.epsilon
		})

		return config


	def call(self, inputs):

		mean = backend.mean(inputs, axis = 0, keepdims = True)
		squ_diffs = backend.square(inputs - mean)
		mean_sq_diff = backend.mean(squ_diffs, axis = 0, keepdims = True) + self.epsilon
		stdev = backend.sqrt(mean_sq_diff)
		mean_pix = backend.mean(stdev, keepdims = True)
		shape = backend.shape(inputs)
		output = backend.tile(mean_pix, (shape[0], shape[1], shape[2], 1))

		return backend.concatenate([inputs, output], axis = -1)
