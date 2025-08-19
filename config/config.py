# Dataset variables
COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
IMG_SIZE = 512
PATCH_SIZE = 16
PROB_AUGMENT_TRAINING = 0.85 # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0 # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3"
FPN_CH = 192
DINO_MODEL = "dinov3_vits16plus"
MODEL_TO_NUM_LAYERS = {
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "inov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
MODEL_TO_EMBED_DIM = {
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "inov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}
N_LAYERS_UNFREEZE = 0

# TRAINING PARAMETERS
BATCH_SIZE = 32 # Batch size
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
WEIGHT_REG = 1.0
WEIGHT_CTR = 1.0

LEARNING_RATE = 0.0001 # Learning rate
NUM_EPOCHS = 15 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/siam_tracking/results/2025-08-18_23-40-30/model_0.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/siam_tracking/results' # Folder where the result will be saved