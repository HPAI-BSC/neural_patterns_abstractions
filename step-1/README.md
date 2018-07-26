## First Step
For this step you need python with nltk installed. This code will select the synsets that have at least 1000 images on the imagenet validation dataset

**Input**: A file with all the names of the images and its correspondent class ( writed as an imagenet synset). 

file: step-1/data/imagenet2012_val_synset_codes.txt

Example of the file data: 

image name___________________image class      
ILSVRC2012_val_00000001.JPEG n01751748
ILSVRC2012_val_00000002.JPEG n09193705
ILSVRC2012_val_00000003.JPEG n02105855 

If you dont have this file, you can generate it with the generate_imagenet_val_synset_codes.py script, located within this same directory.

**Output**:The partitions of all the synsets with more than 1000 images. 

For each synset with name *name* it will create to files in data: 

-  data/all_synset_partitions_npz/synset/*name*_imgs.npz
-  data/all_synset_partitions_npz/no_synset/no_*name*_imgs.npz

Each file containing a dictionary (we will call it *potato* to simplify) such that: 

- potato['imgs'] is a numpy array of strings where every value is the name of an image. The images of this array will change depending on the file, if we loaded the potato_imgs.npz, it will contain the images of *potato* and its hyponims, for example the images of dark potatoes, white potatoes etc. 
 (for example *ILSVRC2012_val_00049973.JPEG*)
- potato['name'] is the name of the synset, for example *dog*.
- potato['code'] is the imagenet code of the synset, for example *n02133161*

The synsets selected using the imagenet_2012 validation dataset are: 
'food', 'structure', 'entity', 'animal', 'aquatic_bird', 'organism', 'solid', 'hunting_dog',
'equipment', 'consumer_goods', 'mammal', 'arthropod', 'chordate', 'living_thing', 'covering',
'carnivore', 'craft', 'invertebrate', 'vertebrate', 'self-propelled_vehicle', 'clothing', 'canine',
'terrier', 'insect', 'implement', 'dog', 'motor_vehicle', 'physical_entity', 'vehicle', 'vessel',
'reptile', 'musical_instrument', 'furniture', 'commodity', 'wheeled_vehicle', 'artifact', 'conveyance',
'domestic_animal', 'primate', 'working_dog', 'object', 'garment', 'bird', 'whole', 'instrument',
'substance', 'matter', 'furnishing', 'container', 'instrumentality', 'diapsid', 'produce',
'protective_covering', 'device', 'placental'
	            
	            
#### Example of use: 
 Modify the paths as liked and run with python. 
 
   
