import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pycocotools.coco import COCO

COCO_ROOT = '/home/rafa/deep_learning/datasets/COCO'

# Initialize COCO API for instance annotations
coco = COCO(os.path.join(COCO_ROOT, 'annotations', 'instances_train2017.json'))

# Load a sample image
img_id = coco.getImgIds()[0]
img_info = coco.loadImgs(img_id)[0]
image_path = os.path.join(COCO_ROOT, 'train2017', img_info['file_name'])
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Load annotations
ann_ids = coco.getAnnIds(imgIds=img_id)
annotations = coco.loadAnns(ann_ids)

# Draw bounding boxes and category names
for ann in annotations:
    x, y, w, h = ann['bbox']
    category_id = ann['category_id']
    category_name = coco.loadCats(category_id)[0]['name']
    
    # Draw rectangle
    cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
    
    # Put category name
    cv2.putText(image, category_name, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

# Display the image with bounding boxes and category names
plt.imshow(image)
plt.axis('off')
plt.show()
