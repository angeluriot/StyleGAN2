import math
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Activation, LeakyReLU, Reshape, UpSampling2D, Cropping2D, Add

from layers import get_filters, EqualizedDense, ModulatedConv2D, AddNoise, Bias


def build_style_block(input, w, noise, filters, kernel_size, alpha, gain):

	style = EqualizedDense(input.shape[-1], bias_init = 1., gain = gain)(w)
	model = ModulatedConv2D(filters, kernel_size, True, gain = gain)([input, style])

	n = Cropping2D((noise.shape[1] - model.shape[1]) // 2)(noise)
	model = AddNoise()([model, n])
	model = Bias()(model)
	model = LeakyReLU(alpha)(model)

	return model


def to_rgb(input, w, nb_channels, gain):

	style = EqualizedDense(input.shape[-1], bias_init = 1., gain = gain)(w)
	model = ModulatedConv2D(nb_channels, 1, False, gain = gain)([input, style])
	model = Bias()(model)

	return model


def build_gen_block(input, w, noise_1, noise_2, filters, nb_channels, kernel_size, alpha, gain):

	model = build_style_block(input, w, noise_1, filters, kernel_size, alpha, gain)
	model = build_style_block(model, w, noise_2, filters, kernel_size, alpha, gain)

	rgb = to_rgb(model, w, nb_channels, gain)

	return model, rgb


def build_generator(latent_dim, image_size, nb_channels, min_image_size, max_filters, min_filters, kernel_size, alpha, gain):

	nb_blocks = int(math.log(image_size, 2)) - int(math.log(min_image_size, 2)) + 1
	filters = get_filters(max_filters, min_filters, nb_blocks, True)

	model_input = Input(shape = (1,))
	w_inputs = [Input(shape = (latent_dim,)) for _ in range(nb_blocks)]
	noise_inputs = [Input(shape = (image_size, image_size, 1)) for _ in range((nb_blocks * 2) - 1)]

	model = EqualizedDense(min_image_size * min_image_size * filters[0], use_bias = False, gain = gain)(model_input)
	model = Reshape((min_image_size, min_image_size, filters[0]))(model)

	model = build_style_block(model, w_inputs[0], noise_inputs[0], filters[0], kernel_size, alpha, gain)
	rgb = to_rgb(model, w_inputs[0], nb_channels, gain)

	for i in range(1, nb_blocks):
		model = UpSampling2D(interpolation = "bilinear")(model)
		model, new_rgb = build_gen_block(model, w_inputs[i], noise_inputs[(i * 2) - 1], noise_inputs[i * 2], filters[i], nb_channels, kernel_size, alpha, gain)
		rgb = Add()([UpSampling2D(interpolation = "bilinear")(rgb), new_rgb])

	rgb = Activation("tanh")(rgb)

	return Model([model_input] + w_inputs + noise_inputs, rgb)
