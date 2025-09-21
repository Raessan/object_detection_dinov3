# Dataset variables
COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO' # Location of dataset
IMG_SIZE = 640 # Size of the image (it will be square)
PATCH_SIZE = 16 # Patch size for the transformer embeddings
PROB_AUGMENT_TRAINING = 0.0 # Probability to perform photogrametric augmentation in training
PROB_AUGMENT_VALID = 0.0 # Probability to perform photogrametric augmentation in validation
IMG_MEAN = [0.485, 0.456, 0.406] # Mean of the image that the backbone (e.g. ResNet) expects
IMG_STD = [0.229, 0.224, 0.225] # Std of the image that the backbone (e.g. ResNet) expects

# Model parameters
DINOV3_DIR = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3" # Directory for dinov3 code
FPN_CH = 192 # Number of channels for the Feature Pyramid Network
DINO_MODEL = "dinov3_vits16plus" # Type of DINOv3 model to use
DINO_WEIGHTS = "/home/rafa/deep_learning/projects/object_detection_dinov3/dinov3_weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth" # Location of weights of DINOv3 model
MODEL_TO_NUM_LAYERS = { # Mapping from model type to number of layers
    "dinov3_vits16": 12,
    "dinov3_vits16plus": 12,
    "dinov3_vitb16": 12,
    "dinov3_vitl16": 24,
    "dinov3_vith16plus": 32,
    "dinov3_vit7b16": 40,
}
MODEL_TO_EMBED_DIM = { # Mapping from model type to embedding dimension
    "dinov3_vits16": 384,
    "dinov3_vits16plus": 384,
    "dinov3_vitb16": 768,
    "dinov3_vitl16": 1024,
    "dinov3_vith16plus": 1280,
    "dinov3_vit7b16": 4096,
}
N_LAYERS_UNFREEZE = 0 # Whether to unfreeze last layers (keep 0 for compatibility with other heads)
N_CONVS = 4 # Number of convolutions

# TRAINING PARAMETERS
BATCH_SIZE = 16 # Batch size
FOCAL_ALPHA = 0.25 # Value of alpha for the focal loss
FOCAL_GAMMA = 2.0 # Value of gamma for the focal loss
WEIGHT_REG = 1.0 # Weight for the regression loss
WEIGHT_CTR = 1.0 # Weight for the centerness loss

LEARNING_RATE = 0.0001 # Learning rate
WEIGHT_DECAY = 0.0001 # Weight decay for regularization
NUM_EPOCHS = 15 # Number of epochs
NUM_SAMPLES_PLOT = 6 # Number of samples to plot during training or validation

LOAD_MODEL = False # Whether to load an existing model for training
SAVE_MODEL = True # Whether to save the result from the training
MODEL_PATH_TRAIN_LOAD = '/home/rafa/deep_learning/projects/object_detection_dinov3/results/2025-08-26_22-52-57/2025-08-27_20-28-53/model_13.pth' # Path of the model to load
RESULTS_PATH = '/home/rafa/deep_learning/projects/object_detection_dinov3/results' # Folder where the result will be saved

# PARAMETERS FOR INFERENCE
MODEL_PATH_INFERENCE = '/home/rafa/deep_learning/projects/object_detection_dinov3/weights/model.pth' # Path of the model to perform inference
IMG_INFERENCE_PATH = '/home/rafa/deep_learning/datasets/COCO/val2017/000000000139.jpg' # Image to perform inference
SCORE_THRESH = 0.2 # Score threshold to accept bounding boxes
NMS_THRESH = 0.6 # Non-max-supression threshold to discard boxes
CLASS_NAMES_PATH = "/home/rafa/deep_learning/projects/object_detection_dinov3/src/class_names.txt" # Path of the file with class names