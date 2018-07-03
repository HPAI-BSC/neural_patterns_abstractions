import os

data_path = '/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/ILSVRC_12_val'
data_labels = '/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/imagenet2012_val_synset_codes.txt'
first_class_images = '/gpfs/projects/bsc28/bsc28535/tiramisu_3.0_dario_scripts/nltk_stuff/mammal_images.txt'
source_target_path = '/gpfs/projects/bsc28/bsc28535/tiramisu_3.0_dario/imgs/imagenet2012_mam/train'

#Store image names corresponding to living things
liv_img = []
with open(first_class_images,'r') as f:
    for l in f:
        liv_img.append(l.strip())

with open(data_labels,'r') as f:
    for l in f:
        img,synset = l.strip().split(' ')
        #Create the symlink
        if img in liv_img:
            os.symlink(os.path.join(data_path,img),os.path.join(source_target_path,'mammal',img))
        else:
            os.symlink(os.path.join(data_path,img),os.path.join(source_target_path,'nomamm',img))
