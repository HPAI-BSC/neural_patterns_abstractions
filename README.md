# Semantic Transfer

In this repository we will perform an analysis about the relation between the embedding of some synsets and their semantic relationships. 

We use the next resources: 
- The Imagenet 2012 validation dataset (1000 clases, 50k images)
- The Full Network Embedding
- Tiramisu 

We do several steps to generate the necesary data for this analysis. For this reason the repository is divided in one folder per each step. 

### First Step

**Input**: A file with all the names of the images and its correspondent class (notation: imagenet synset). 

step-1/data/imagenet2012_val_synset_codes.txt

**Output**: The files with the discriminative partition of the classes with at least 1000 images (including hyponims). 

step-1/data/synset/synset_imgs.txt
step-1/data/synset/no_synset_imgs.txt
