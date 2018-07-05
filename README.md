# Semantic Transfer

In this repository we will perform an analysis about the relation between the embedding of some synsets and their semantic relationships. 

We use the next resources: 
- The Imagenet 2012 validation dataset (1000 clases, 50k images)
- The Full Network Embedding
- Tiramisu 

We do several steps to generate the necesary data for this analysis. For this reason the repository is divided in one folder per each step. 

### First Step
For this step you need a python with nltk installed. 

**Input**: A file with all the names of the images and its correspondent class ( writed as an imagenet synset). 

file: step-1/data/imagenet2012_val_synset_codes.txt

**Output**:The partitions of all the synsets with more than 1000 images.

For each synset it will create to files in data: 

-  data/all_synset_partitions_npz/synset/name_imgs.npz
-  data/all_synset_partitions_npz/no_synset/no_name_imgs.npz


### Second Step
This code is in:  /gpfs/projects/bsc28/semantic_transfer_scripts/code

First we generate a tiramisu folder. TODO: explain this step. 

**Input**: The files generated on the step 1 of the folder data/all_synset_partitions_npz/synset/ . There are 55 files. 


**Output**:The folders with the symbolic links on tiramisu_nicename/imgs for every one of the selected synsets. 

The structure of this will be supposing the synset is dog: 

- tiramisu_nicename/imgs/dog/train/dog/images0008.JPEG

- tiramisu_nicename/imgs/dog/train/no_dog/images00058.JPEG


### Third Step
This code is in:  /gpfs/projects/bsc28/tiramisu_semantic_transfer

The main objective of this step is to create all the embeddings for the selected embeddings. To do so, we will use the tiramisu3.0 code on the minotauro.

**Input**:  The folders with the symbolic links on tiramisu_nicename/imgs for every one of the selected synsets. 


**Output**: One embedding per synset selected. 