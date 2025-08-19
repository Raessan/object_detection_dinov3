import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pycocotools.coco import COCO
import cv2
import numpy as np
import random
import os
from src.utils import image_to_tensor, resize_transform, tensor_to_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class DatasetCOCO(Dataset):
    def __init__(self, root_dir, mode, img_size, patch_size, augment_prob=0.8, 
                 mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the annotation file.
            transform (callable, optional): Optional transform to be applied on a sample.
            augment_prob (float): Probability of applying augmentations.
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size
        if self.mode == "train":
            self.data_dir = os.path.join(self.root_dir, "train2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "instances_train2017.json")
        elif self.mode == "val":
            self.data_dir = os.path.join(self.root_dir, "val2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "instances_val2017.json")
        elif self.mode == "test":
            self.data_dir = os.path.join(self.root_dir, "test2017")
            self.path_annotations = os.path.join(self.root_dir, "annotations", "instances_test2017.json")
        else:
            raise Exception("the mode must be either train, val or test")
        self.coco = COCO(self.path_annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augment_prob = augment_prob
        self.mean = np.array(mean, dtype=np.float32)[None, :, None, None]
        self.std  = np.array(std, dtype=np.float32)[None, :, None, None]

        cat_ids = self.coco.getCatIds()
        cats = self.coco.loadCats(cat_ids)
        self.class_names = [cat['name'] for cat in cats]
        #print("COCO classes:", self.class_names)
        print("Total number of classes:", len(self.class_names))

        random.seed(42)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.data_dir, img_info['file_name'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []

        # Relative bboxes
        h = image.shape[0]
        w = image.shape[1]

        for ann in annotations:
            x_bbox, y_bbox, w_bbox, h_bbox = ann['bbox']
            boxes.append([x_bbox/w, y_bbox/h, w_bbox/w, h_bbox/h])
            category_id = ann['category_id']
            category_name = self.coco.loadCats(category_id)[0]['name']
            label = self.class_names.index(category_name)
            labels.append(label)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        
        #target = {'boxes': boxes, 'labels': labels}

        # Augment image
        if random.random() < self.augment_prob:
            image = self.photometric_augment(image)

        # Resize image
        image = resize_transform(image, self.img_size, self.patch_size)

        return image_to_tensor(image, self.mean, self.std), boxes, labels

    def photometric_augment(self, img,
                        brightness_delta=32,
                        contrast_range=(0.8,1.2),
                        saturation_range=(0.8,1.2),
                        hue_delta=10,
                        p_jitter=0.5,
                        p_blur=0.3,
                        p_noise=0.3):
        """
        img: H×W×3 BGR uint8
        Returns: same shape uint8
        """
        out = img.astype(np.float32)

        # 1) brightness
        if random.random() < p_jitter:
            delta = random.uniform(-brightness_delta, brightness_delta)
            out += delta

        # 2) contrast
        if random.random() < p_jitter:
            alpha = random.uniform(*contrast_range)
            out *= alpha

        # 3) saturation & hue (convert to HSV in OpenCV)
        if random.random() < p_jitter:
            hsv = cv2.cvtColor(out.clip(0,255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[...,1] *= random.uniform(*saturation_range)  # sat
            hsv[...,0] += random.uniform(-hue_delta, hue_delta)  # hue
            out = cv2.cvtColor(hsv.clip(0,255).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)

        # 4) Gaussian blur
        if random.random() < p_blur:
            k = random.choice([3,5])  # odd kernel size
            out = cv2.GaussianBlur(out, (k,k), 0)

        # 5) Additive Gaussian noise
        if random.random() < p_noise:
            sigma = random.uniform(5, 20)
            noise = np.random.randn(*out.shape) * sigma
            out += noise

        return out.clip(0,255).astype(np.uint8)
    
    def visualize(self, idx):
        """
        Visualizes an image and its bounding boxes with category labels.

        Args:
            idx (int): Index of the image in the dataset.
        """
        image_tensor, boxes, labels = self.__getitem__(idx)
        image = tensor_to_image(image_tensor, self.mean, self.std)

        # Get absolute bboxes
        h = image.shape[0]
        w = image.shape[1]
        
        for bbox in boxes:
            bbox[0] *= w
            bbox[1] *= h
            bbox[2] *= w
            bbox[3] *= h

        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        for box, label in zip(boxes, labels):
            x, y, w, h = box.tolist()
            rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            category_name = self.class_names[label.item()] #self.coco.loadCats(label.item())[0]['name']
            ax.text(x, y - 10, category_name, color='red', fontsize=10)

        plt.axis('off')
        plt.show()
    
if __name__ == '__main__':
    COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'
    MODE = "val"
    IMG_SIZE = 768
    PATCH_SIZE = 16
    dataset = DatasetCOCO(COCO_ROOT, MODE, IMG_SIZE, PATCH_SIZE)
    data = dataset.__getitem__(0)
    dataset.visualize(0)