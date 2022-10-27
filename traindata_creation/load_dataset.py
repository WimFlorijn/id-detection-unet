import os
import midv500


dataset_name = "midv2019"

file_dir = os.path.dirname(os.path.realpath(__file__))
target_dir = os.path.join(file_dir, '..', 'dataset')
if not os.path.exists(target_dir):
    os.mkdir(target_dir)

midv500.download_dataset(target_dir, dataset_name)


