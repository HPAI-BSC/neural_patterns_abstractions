import os


data_path = '/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/ILSVRC_12_val'
data_labels = '/gpfs/projects/bsc28/DATASETS/Imagenet_val50k/imagenet2012_val_synset_codes.txt'
source_target_path = '/gpfs/projects/bsc28/bsc28535/tiramisu_3.0_dario/imgs/imagenet2012/train'

with open(data_labels,'r') as f:
    for l in f:
        img,synset = l.strip().split(' ')
        #Create the folder
        if not os.path.exists(os.path.join(source_target_path,synset)):
            os.makedirs(os.path.join(source_target_path,synset))
        #Create the symlink
        os.symlink(os.path.join(data_path,img),os.path.join(source_target_path,synset,img))
