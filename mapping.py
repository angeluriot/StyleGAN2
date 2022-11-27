from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU

from layers import PixelNorm, EqualizedDense

def build_mapping(latent_dim, mapping_layers, mapping_lr_ratio, alpha, gain):

	model_input = Input(shape = (latent_dim,))
	model = PixelNorm()(model_input)

	for _ in range(mapping_layers):
		model = EqualizedDense(latent_dim, gain = gain, lr_multiplier = mapping_lr_ratio)(model)
		model = LeakyReLU(alpha)(model)

	return Model(model_input, model)
