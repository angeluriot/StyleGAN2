import os
import math

# Dataset
DATA_DIR = "E:/Angel/Development/Datasets/anime_faces_128"
IMAGE_SIZE = 128
NB_CHANNELS = 3
FLIP_DATASET = True

# Outputs
OUTPUT_DIR = "output"
SAMPLES_DIR = "images"
MODELS_DIR = "models"
OUTPUT_SHAPE = (7, 8)
MARGIN = IMAGE_SIZE // 8
SAVE_PER_EPOCH = 20

# Model
LATENT_DIM = 512
MAPPING_LAYERS = 8
MIN_IMAGE_SIZE = 4
MAX_FILTERS = 512
MIN_FILTERS = 64
KERNEL_SIZE = 3
ALPHA = 0.2
GAIN = 1.3

# Training
BATCH_SIZE = 8
NB_EPOCHS = 1000
MAX_LR = 0.0012
MIN_LR = 0.0005
LR_SPEED = 0.03
MAPPING_LR_RATIO = 0.01
BETA_1 = 0.
BETA_2 = 0.99
EPSILON = 10e-8
STYLE_MIX_PROBA = 0.5
GRADIENT_PENALTY_COEF = 10.
GRADIENT_PENALTY_INTERVAL = 4
MA_HALF_LIFE = 10.

# Calculated
NB_DATA = len(os.listdir(DATA_DIR)) * 2 if FLIP_DATASET else len(os.listdir(DATA_DIR))
NB_BATCHS = math.ceil(float(NB_DATA) / float(BATCH_SIZE))
SAMPLES_DIR = os.path.join(OUTPUT_DIR, SAMPLES_DIR)
MODELS_DIR = os.path.join(OUTPUT_DIR, MODELS_DIR)
MA_BETA = 0.5 ** (BATCH_SIZE / (MA_HALF_LIFE * 1000.)) if MA_HALF_LIFE > 0. else 0.
NB_BLOCKS = int(math.log(IMAGE_SIZE, 2)) - int(math.log(MIN_IMAGE_SIZE, 2)) + 1
