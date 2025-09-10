import torch

def collate_fn(batch):
    """
    batch: list of tuples (image_tensor, boxes, labels)
    """
    # images, boxes, labels = zip(*batch)
    # # keep targets as a list of dicts (variable-length bboxes)
    # return images, boxes, labels

    images = [item[0] for item in batch]
    bboxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    # stack images into one tensor
    images = torch.stack(images, dim=0)
    # keep targets as a list of dicts (variable-length bboxes)
    return images, bboxes, labels