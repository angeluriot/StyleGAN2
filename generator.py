from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation, LeakyReLU, Reshape, UpSampling2D, Cropping2D, Add

from settings import *
from layers import *


def build_style_block(input, w, noise, filters):

	style = EqualizedDense(input.shape[-1], bias_init = 1.)(w)
	model = ModulatedConv2D(filters, KERNEL_SIZE, True)([input, style])

	n = Cropping2D((noise.shape[1] - model.shape[1]) // 2)(noise)
	model = AddNoise()([model, n])
	model = Bias()(model)
	model = LeakyReLU(ALPHA)(model)

	return model


def to_rgb(input, w):

	style = EqualizedDense(input.shape[-1], bias_init = 1.)(w)
	model = ModulatedConv2D(NB_CHANNELS, 1, False)([input, style])
	model = Bias()(model)

	return model


def build_block(input, w, noise_1, noise_2, filters):

	model = build_style_block(input, w, noise_1, filters)
	model = build_style_block(model, w, noise_2, filters)

	rgb = to_rgb(model, w)

	return model, rgb


def build_model():

	filters = get_filters(True)

	model_input = Input(shape = (1,))
	w_inputs = [Input(shape = (LATENT_DIM,)) for _ in range(NB_BLOCKS)]
	noise_inputs = [Input(shape = (IMAGE_SIZE, IMAGE_SIZE, 1)) for _ in range((NB_BLOCKS * 2) - 1)]

	model = EqualizedDense(MIN_IMAGE_SIZE * MIN_IMAGE_SIZE * filters[0], use_bias = False)(model_input)
	model = Reshape((MIN_IMAGE_SIZE, MIN_IMAGE_SIZE, filters[0]))(model)

	model = build_style_block(model, w_inputs[0], noise_inputs[0], filters[0])
	rgb = to_rgb(model, w_inputs[0])

	for i in range(1, NB_BLOCKS):
		model = UpSampling2D(interpolation = "bilinear")(model)
		model, new_rgb = build_block(model, w_inputs[i], noise_inputs[(i * 2) - 1], noise_inputs[i * 2], filters[i])
		rgb = Add()([UpSampling2D(interpolation = "bilinear")(rgb), new_rgb])

	return Model([model_input] + w_inputs + noise_inputs, rgb)
