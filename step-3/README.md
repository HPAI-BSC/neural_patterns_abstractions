### Third step
In this step we use the Tiramisu implementation of the FNE. We will call the folder where is all this implementation fne\_paco. 

To run it, you need to put the env_raquel.sh file in the enviroment\_variables of fne\_paco and the other two files 
(driver\_extractor.py and embedding\_reader.py) on the root of fne\_paco. Then, once the fne\_paco is installed on marenostrum (following the instructons of fne\_paco/README.md) 
you call (from the root of fne\_paco): 

python3 driver\_extractor.py
 

It will generate all the embeddings of the selected synsets, and will extract the features. 