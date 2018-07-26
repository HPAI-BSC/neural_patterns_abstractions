'''
The generate_partition.py file requires one file with the images and their synset label.
This code generates such file, using the freely available 
'''

numeric_labels_path = '../../data/val_numeric_labels.txt'
numeric_synset_path = '../../data/imagenet_lsvrc_2012_synsets.txt'


number_to_synset = {}
with open(numeric_synset_path,'r') as f:
    counter = 0
    for l in f:
        number_to_synset[counter]=l.strip()
        counter+=1

output = open('../../data/imagenet2012_val_synset_codes.txt','w')

with open(numeric_labels_path,'r') as f:
    for l in f:
        image,label = l.strip().split(' ')
        output.write(image+' '+number_to_synset[int(label)]+'\n')
