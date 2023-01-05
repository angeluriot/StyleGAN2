import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU, Add, Lambda, AveragePooling2D, Flatten

from settings import *
from layers import *


def from_rgb(input, filters):

	model = EqualizedConv2D(filters, 1)(input)
	model = LeakyReLU(ALPHA)(model)

	return model


def build_block(input, filters):

	residual = EqualizedConv2D(filters, 1, use_bias = False)(input)
	residual = AveragePooling2D()(residual)

	model = EqualizedConv2D(filters, KERNEL_SIZE)(input)
	model = LeakyReLU(ALPHA)(model)

	model = EqualizedConv2D(filters, KERNEL_SIZE)(model)
	model = LeakyReLU(ALPHA)(model)

	model = AveragePooling2D()(model)
	model = Add()([model, residual])
	model = Lambda(lambda x: x / math.sqrt(2.))(model)

	return model


def build_model():

	filters = get_filters(False)

	model_input = Input(shape = (IMAGE_SIZE, IMAGE_SIZE, NB_CHANNELS))

	model = from_rgb(model_input, filters[0])

	for i in range(NB_BLOCKS - 1):
		model = build_block(model, filters[i])

	model = MinibatchStdDev()(model)

	model = EqualizedConv2D(filters[-1], KERNEL_SIZE)(model)
	model = LeakyReLU(ALPHA)(model)

	model = Flatten()(model)

	model = EqualizedDense(filters[-1])(model)
	model = LeakyReLU(ALPHA)(model)

	model = EqualizedDense(1)(model)

	return Model(model_input, model)

