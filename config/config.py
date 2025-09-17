# Dataset variables
COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
IMG_SIZE = 640
PATCH_SIZE = 16
PROB_AUGMENT_TRAINING = 0.0 # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0 # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3"
FPN_CH = 192
DINO_MODEL = "dinov3_vits16plus"
DINO_WEIGHTS = "/home/rafa/deep_learning/projects/object_detection_dinov3/results/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"
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
N_CONVS = 4

# TRAINING PARAMETERS
BATCH_SIZE = 16 # Batch size
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0
WEIGHT_REG = 1.0
WEIGHT_CTR = 1.0

LEARNING_RATE = 0.0001 # Learning rate
WEIGHT_DECAY = 0.0001
NUM_EPOCHS = 15 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/object_detection_dinov3/results/2025-08-26_22-52-57/2025-08-27_20-28-53/model_13.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/object_detection_dinov3/results' # Folder where the result will be saved

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/object_detection_dinov3/results/2025-08-26_22-52-57/2025-08-27_20-28-53/model_13.pth' # Path of the model to perform inference
IMG_INFERENCE_PATH = '/home/rafa/deep_learning/datasets/COCO/val2017/000000000139.jpg'
SCORE_THRESH = 0.2
NMS_THRESH = 0.6