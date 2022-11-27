import os
import math
import numpy as np
import gc
from PIL import Image
from tensorflow.keras.callbacks import Callback
import tensorflow as tf
from tensorflow.keras import backend

import utils


class UpdateVariables(Callback):

	def __init__(self, **kwargs):

		super(Callback, self).__init__(**kwargs)


	def on_batch_begin(self, batch, logs = None):

		self.model.batch.assign(batch)


	def on_batch_end(self, batch, logs = None):

		self.model.moving_average()


	def on_epoch_begin(self, epoch, logs = None):

		self.model.epoch.assign(epoch)


class SaveSamples(Callback):

	def __init__(self, z, noise, batch_size, margin, shape, output_dir, samples_dir, save_per_epoch, nb_batchs, **kwargs):

		super(Callback, self).__init__(**kwargs)
		self.z = z
		self.noise = noise
		self.batch_size = batch_size
		self.margin = margin
		self.shape = shape
		self.output_dir = output_dir
		self.samples_dir = samples_dir
		self.epoch = 0
		self.save_rate = math.ceil(nb_batchs / save_per_epoch)


	def on_batch_end(self, batch, logs = None):

		if batch % self.save_rate == 0:

			generations = self.model.predict(self.z, list(self.noise), self.batch_size)

			output_image = np.full((
				self.margin + (self.shape[1] * (generations.shape[2] + self.margin)),
				self.margin + (self.shape[0] * (generations.shape[1] + self.margin)),
				generations.shape[3]), 255, dtype = np.uint8
			)

			i = 0
			for row in range(self.shape[1]):
				for col in range(self.shape[0]):
					r = row * (generations.shape[2] + self.margin) + self.margin
					c = col * (generations.shape[1] + self.margin) + self.margin
					output_image[r:r + generations.shape[2], c:c + generations.shape[1]] = generations[i]
					i += 1

			if not os.path.exists(self.samples_dir):
				os.makedirs(self.samples_dir)

			img = Image.fromarray(output_image)
			img.save(os.path.join(self.samples_dir, "image_" + str(self.epoch + 1) + "_" + str(batch + 1) + ".png"))
			img.save(os.path.join(self.output_dir, "last_image.png"))


	def on_epoch_begin(self, epoch, logs = None):

		self.epoch = epoch


class SaveModels(Callback):

	def __init__(self, output_dir, **kwargs):

		super(Callback, self).__init__(**kwargs)
		self.output_dir = output_dir


	def on_epoch_end(self, epoch, logs = None):

		self.model.save_weights(self.output_dir, epoch + 1)
