FOLDER_NAME = 'cogniplan_nav_pred7'
model_path = f'checkpoints/{FOLDER_NAME}'
gifs_path = f'{model_path}/results/gifs'
generator_path = 'checkpoints/wgan_inpainting'
N_GEN_SAMPLE = 4
trajectory_path = f'{model_path}/results/trajectory'
length_path = f'{model_path}/results/length'

INPUT_DIM = 9
EMBEDDING_DIM = 128
K_SIZE = 20  # the number of neighbors
USE_GPU = False  # do you want to use GPUS?
NUM_GPU = 1 # the number of GPUs
NUM_META_AGENT = 10  # the number of processes
NUM_TEST = 100
SAVE_GIFS = False  # do you want to save GIFs
SAVE_TRAJECTORY = False  # do you want to save per-step metrics
SAVE_LENGTH = False  # do you want to save per-episode metrics
