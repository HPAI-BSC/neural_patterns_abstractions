# Semantic Transfer

In this repository we will perform an analysis about the relation between the embedding of some synsets and their semantic relationships. 

We use the next resources: 
- The Imagenet 2012 validation dataset (1000 clases, 50k images)
- The Full Network Embedding
- Tiramisu 

We do several steps to generate the necesary data for this analysis. For this reason the repository is divided in one folder per each step. 

### First Step

**Input**: A file with all the names of the images and its correspondent class ( writed as an imagenet synset). 

file: step-1/data/imagenet2012_val_synset_codes.txt

**Output**:The partitions of all the synsets with more than 1000 images.
For each synset it will create a folder in data, and inside this folder it will write the partition files: 
  step-1/data/synset/synset_imgs.txt
  step-1/data/synset/no_synset_imgs.txt
