import os
import math
import numpy as np
from PIL import Image
from tensorflow.keras.callbacks import Callback
from keras import backend

from settings import *

class Updates(Callback):

	def __init__(self, **kwargs):

		super(Callback, self).__init__(**kwargs)


	def on_batch_begin(self, batch, logs = None):

		self.model.batch.assign(batch)


	def on_batch_end(self, batch, logs = None):

		self.model.moving_average()


	def on_epoch_begin(self, epoch, logs = None):

		self.model.epoch.assign(epoch)

		lr = MIN_LR + (MAX_LR - MIN_LR) * np.exp(-float(epoch) * LR_SPEED)
		backend.set_value(self.model.generator_optimizer.learning_rate, lr)
		backend.set_value(self.model.discriminator_optimizer.learning_rate, lr)


class SaveSamples(Callback):

	def __init__(self, z, noise, **kwargs):

		super(Callback, self).__init__(**kwargs)
		self.z = z
		self.noise = noise
		self.epoch = 0
		self.save_rate = math.ceil(float(NB_BATCHS) / float(SAVE_PER_EPOCH))


	def on_batch_end(self, batch, logs = None):

		if batch % self.save_rate == 0:

			generations = self.model.predict(self.z, list(self.noise))

			output_image = np.full((
				MARGIN + (OUTPUT_SHAPE[1] * (generations.shape[2] + MARGIN)),
				MARGIN + (OUTPUT_SHAPE[0] * (generations.shape[1] + MARGIN)),
				generations.shape[3]), 255, dtype = np.uint8
			)

			i = 0

			for row in range(OUTPUT_SHAPE[1]):
				for col in range(OUTPUT_SHAPE[0]):
					r = row * (generations.shape[2] + MARGIN) + MARGIN
					c = col * (generations.shape[1] + MARGIN) + MARGIN
					output_image[r:r + generations.shape[2], c:c + generations.shape[1]] = generations[i]
					i += 1

			if not os.path.exists(SAMPLES_DIR):
				os.makedirs(SAMPLES_DIR)

			img = Image.fromarray(output_image)
			img.save(os.path.join(SAMPLES_DIR, "image_" + str(self.epoch + 1) + "_" + str(batch + 1) + ".png"))
			img.save(os.path.join(OUTPUT_DIR, "last_image.png"))


	def on_epoch_begin(self, epoch, logs = None):

		self.epoch = epoch


class SaveModels(Callback):

	def __init__(self, **kwargs):

		super(Callback, self).__init__(**kwargs)


	def on_epoch_end(self, epoch, logs = None):

		self.model.save_weights(MODELS_DIR, epoch + 1)
