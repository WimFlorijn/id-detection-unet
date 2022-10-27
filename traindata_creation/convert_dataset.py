import os
import cv2
import json
import numpy as np

from tqdm import tqdm
from midv500.utils import list_annotation_paths_recursively, create_dir


file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(file_dir, '..', 'dataset')

export_dir = os.path.join(file_dir, '..', 'traindata', 'data')
mask_dir = os.path.join(export_dir, 'masks')
image_dir = os.path.join(export_dir, 'images')

for directory in (export_dir, mask_dir, image_dir):
    create_dir(directory)


TARGET_SHAPE = (1664, 1024)

annotation_paths = list_annotation_paths_recursively(root_dir)
for ind, rel_annotation_path in enumerate(tqdm(annotation_paths)):
    # get image path
    rel_image_path = rel_annotation_path.replace("ground_truth", "images")
    rel_image_path = rel_image_path.replace("json", "tif")

    # load image
    abs_image_path = os.path.join(root_dir, rel_image_path)
    image = cv2.imread(abs_image_path)

    # load mask coords
    abs_annotation_path = os.path.join(root_dir, rel_annotation_path)
    quad = json.load(open(abs_annotation_path, "r"))
    mask_coords = quad["quad"]

    # create mask from poly coords
    mask = np.zeros(image.shape, dtype=np.uint8)
    mask_coords_np = np.array(mask_coords, dtype=np.int32)
    cv2.fillPoly(mask, mask_coords_np.reshape(-1, 4, 2), color=(255, 255, 255))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)[1]

    image = cv2.resize(image, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)
    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)

    cv2.imwrite(os.path.join(image_dir, f'{ind}.tiff'), image)
    cv2.imwrite(os.path.join(mask_dir, f'{ind}.tiff'), mask)
