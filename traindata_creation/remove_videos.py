import os
import shutil

file_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.join(file_dir, '..', 'dataset')

for directory in os.listdir(root_dir):
    for directory_1 in os.listdir(os.path.join(root_dir, directory)):
        if os.path.exists(os.path.join(root_dir, directory, directory_1, 'videos')):
            shutil.rmtree(os.path.join(root_dir, directory, directory_1, 'videos'))
