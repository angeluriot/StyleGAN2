import math
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, Add, Lambda, AveragePooling2D, Flatten, Activation

from layers import get_filters, EqualizedDense, EqualizedConv2D, MinibatchStdDev


def from_rgb(input, filters, alpha, gain):

	model = EqualizedConv2D(filters, 1, gain = gain)(input)
	model = LeakyReLU(alpha)(model)

	return model


def build_dis_block(input, filters, kernel_size, alpha, gain):

	residual = EqualizedConv2D(filters, 1, use_bias = False, gain = gain)(input)
	residual = AveragePooling2D()(residual)

	model = EqualizedConv2D(filters, kernel_size, gain = gain)(input)
	model = LeakyReLU(alpha)(model)

	model = EqualizedConv2D(filters, kernel_size, gain = gain)(model)
	model = LeakyReLU(alpha)(model)

	model = AveragePooling2D()(model)
	model = Add()([model, residual])
	model = Lambda(lambda x: x / math.sqrt(2.))(model)

	return model


def build_discriminator(image_size, nb_channels, min_image_size, max_filters, min_filters, kernel_size, alpha, gain):

	nb_blocks = int(math.log(image_size, 2)) - int(math.log(min_image_size, 2)) + 1
	filters = get_filters(max_filters, min_filters, nb_blocks, False)

	model_input = Input(shape = (image_size, image_size, nb_channels))

	model = from_rgb(model_input, filters[0], alpha, gain)

	for i in range(nb_blocks - 1):
		model = build_dis_block(model, filters[i], kernel_size, alpha, gain)

	model = MinibatchStdDev()(model)

	model = EqualizedConv2D(filters[-1], kernel_size, gain = gain)(model)
	model = LeakyReLU(alpha)(model)

	model = Flatten()(model)

	model = EqualizedDense(filters[-1], gain = gain)(model)
	model = LeakyReLU(alpha)(model)

	model = EqualizedDense(1, gain = gain)(model)

	return Model(model_input, model)

