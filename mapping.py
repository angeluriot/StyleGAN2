from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LeakyReLU

from settings import *
from layers import *

def build_model():

	model_input = Input(shape = (LATENT_DIM,))
	model = PixelNorm()(model_input)

	for _ in range(MAPPING_LAYERS):
		model = EqualizedDense(LATENT_DIM, lr_multiplier = MAPPING_LR_RATIO)(model)
		model = LeakyReLU(ALPHA)(model)

	return Model(model_input, model)
