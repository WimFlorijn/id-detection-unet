import os
import cv2
import json
import zipfile
import logging
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from pathlib import Path

GENERATE_LINE_MASKS = True
GENERATE_BORDER_MASKS = False
GENERATE_BUILDING_MASKS = False

THICKNESS = 8
TARGET_SHAPE = (1664, 1024)
IN_DIR, OUT_DIR = 'D:/verwerkt', os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'traindata')

data_dir = os.path.join(OUT_DIR, 'data')
masks_dir = os.path.join(data_dir, 'masks')
images_dir = os.path.join(data_dir, 'images')
Path(masks_dir).mkdir(parents=True, exist_ok=True)
Path(images_dir).mkdir(parents=True, exist_ok=True)


def generate_line_mask(lines, mask_shape):
    mask = np.zeros(mask_shape, dtype=np.uint8)
    for line in lines:
        cv2.line(mask, (int(line[0][0]), int(line[0][1])), (int(line[1][0]), int(line[1][1])), 255,
                 thickness=THICKNESS, lineType=cv2.LINE_8)
    mask = cv2.resize(mask, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)
    mask = np.nonzero(mask)

    return mask


def generate_lines_from_json(obs, image_shape, attachment):
    lines, points_dict = obs['lines'], obs['points']
    lines = {k: v for k, v in lines.items() if v.get('attachment') == attachment}

    segments = []
    for line, details in lines.items():
        points = details['points']
        for start, stop in zip(points, points[1:]):
            segments.append([points_dict[start]['position'], points_dict[stop]['position']])

    return generate_line_mask(segments, image_shape)


def generate_borders_from_json(obs, image_shape, attachment):
    semantic_lines, points_dict = obs['semantic_lines'], obs['points']
    semantic_lines = {k: v for k, v in semantic_lines.items() if v.get('attachment') == attachment}

    segments = []
    for line, details in semantic_lines.items():
        points = details['points']
        for start, stop in zip(points, points[1:]):
            segments.append([points_dict[start]['position'], points_dict[stop]['position']])

    return generate_line_mask(segments, image_shape)


def generate_building_from_json(obs, image_shape, attachment):
    points_dict, buildings = obs['points'], obs['buildings']
    buildings = {k: v for k, v in buildings.items() if v.get('attachment') == attachment}

    g = nx.Graph()
    g.add_nodes_from(obs['points'].keys())
    for line in obs['lines'].values():
        line_points = line['points']
        for i in range(1, len(line_points)):
            g.add_edge(line_points[i - 1], line_points[i])

    segments = []
    for building in buildings.values():
        building = building['points']
        for i in range(1, len(building)):
            try:
                path = nx.shortest_path(g, building[i - 1], building[i])
                for k in range(1, len(path)):
                    segments.append([path[k - 1], path[k]])
            except nx.exception.NetworkXNoPath:
                segments.append([building[i - 1], building[i]])

    segments = [[points_dict[segment[0]]['position'], points_dict[segment[1]]['position']] for segment in segments]

    return generate_line_mask(segments, image_shape)


def get_masked(img, line_masks, building_masks, new_masks, visualize=False):
    if visualize:
        visualization_image = img.copy()
        if line_masks is not None:
            visualization_image[line_masks] = [255, 0, 0]
        if building_masks is not None:
            visualization_image[building_masks] = [0, 255, 0]
        if new_masks is not None:
            visualization_image[new_masks] = [0, 0, 255]

        plt.figure(figsize=(8, 8))
        plt.imshow(np.hstack((visualization_image, img)))
        plt.show()

    mask_img = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    if line_masks is not None:
        mask_img[line_masks[0], line_masks[1]] = 255
    if building_masks is not None:
        mask_img[building_masks[0], building_masks[1]] = 255
    if new_masks is not None:
        mask_img[new_masks[0], new_masks[1]] = 255

    return mask_img


def get_attachment_name(x):
    return x.split('/')[2]


def read_zip(zip_name):
    with zipfile.ZipFile(zip_name, 'r') as archive:
        prefix, postfix = 'observations/snapshots/latest/', '.latest.json'
        sketch_files = [x for x in archive.namelist() if x.startswith(prefix) and x.endswith(postfix)]
        for i, sketch_file in enumerate(sketch_files):
            sketch_name = sketch_file[len(prefix):-len(postfix)]

            attachment_prefix = 'observations/attachments/'
            image_files = [
                file_name for file_name in archive.namelist()
                if (file_name.startswith(attachment_prefix)
                    and sketch_name in file_name and 'segmentation' not in file_name)
            ]

            if image_files:
                logging.info(f'Processing sketch: {i}: {sketch_name}.')

                with archive.open(sketch_file, 'r') as afh:
                    json_data = json.loads(afh.read())

                attachment_to_image = {
                    get_attachment_name(x): cv2.imdecode(np.frombuffer(archive.read(x), np.uint8), 1)
                    for x in image_files
                }

                for attachment, image in attachment_to_image.items():
                    image_shape = image.shape[:2]
                    image = cv2.resize(image, (TARGET_SHAPE[1], TARGET_SHAPE[0])).astype(np.uint8)

                    sketch_image_name = f'{sketch_name}_{attachment}.tif'

                    if GENERATE_LINE_MASKS:
                        line_masks = generate_lines_from_json(json_data, image_shape, attachment)
                    else:
                        line_masks = None

                    if GENERATE_BORDER_MASKS:
                        border_masks = generate_borders_from_json(json_data, image_shape, attachment)
                    else:
                        border_masks = None

                    if GENERATE_BUILDING_MASKS:
                        building_masks = generate_building_from_json(json_data, image_shape, attachment)
                    else:
                        building_masks = None

                    if line_masks or building_masks or border_masks:
                        masked_img = get_masked(image, line_masks, building_masks, border_masks, visualize=False)

                        cv2.imwrite(os.path.join(images_dir, sketch_image_name), image)
                        cv2.imwrite(os.path.join(masks_dir, sketch_image_name), masked_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s;%(levelname)s;%(message)s')

    for j, name in enumerate(os.listdir(IN_DIR)):
        logging.info(f'Processing project: {j}: {name}.')
        read_zip(os.path.join(IN_DIR, name))
