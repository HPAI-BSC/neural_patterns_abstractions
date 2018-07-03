import os
import numpy as np
from datetime import timedelta
import time


# global paths
tiramisu_base_dir = '/gpfs/projects/bsc28/tiramisu_semantic_transfer/'
base_imgs_path = tiramisu_base_dir + 'imgs/'

data_path = '/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/ILSVRC_12_val'

imgs = np.load('/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/imagenet2012_val_synset_codes_as_array.npz')['imgs']
synsets = np.load('/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/imagenet2012_val_synset_codes_as_array.npz')['ss']


def create_link(synset_data):

	try:
		os.makedirs(base_imgs_path + synset_data['name'] + '/train')
	except:
		print('peta')

	source_target_path = base_imgs_path + synset_data['name'] + '/train'

	for i in range(len(imgs)):
		img, synset = imgs[i] , synsets[i]
		if img in synset_data['imgs']:
			os.symlink(os.path.join(data_path, img), os.path.join(source_target_path, synset_data['name'], img))
		else:
			os.symlink(os.path.join(data_path, img), os.path.join(source_target_path, 'no_' + synset_data['name'], img))


def create_folders(image_file_paths):
	synset_files = list(os.walk(image_file_paths))[0][2]
	for file in synset_files:
		ss_data = np.load(image_file_paths + file)
		create_link(ss_data)

def main():
	image_file_paths = '/gpfs/projects/bsc28/tiramisu_semantic_transfer/synset_partitions/'
	create_folders(image_file_paths)


if __name__ == '__main__':
	init = time.time()
	main()
	print('time:', timedelta(seconds=time.time() - init))
