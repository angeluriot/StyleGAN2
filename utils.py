import numpy as np
import datetime as dt
import tensorflow as tf


def reset_rand():

	now = dt.datetime.now()
	seconds_since_midnight = int((now - now.replace(hour = 0, minute = 0, second = 0, microsecond = 0)).total_seconds())
	tf.random.set_seed(seconds_since_midnight)


def norm_img(img):

	return np.clip((img.astype(np.float32) / 127.5) - 1., -1., 1.)


def tf_norm_img(img):

	return tf.clip_by_value((tf.cast(img, tf.float32) / 127.5) - 1., -1., 1.)


def denorm_img(img):

	return ((np.clip(img, -1., 1.) + 1.) * 127.5).astype(np.uint8)


def tf_denorm_img(img):

	return tf.cast((tf.clip_by_value(img, -1., 1.) + 1.) * 127.5, tf.uint8)
