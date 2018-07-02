import nltk
from nltk.corpus import wordnet as wn

def _recurse_all_hypernyms(synset, all_hypernyms):
    synset_hypernyms = synset.hypernyms()
    if synset_hypernyms:
        all_hypernyms += synset_hypernyms
        for hypernym in synset_hypernyms:
            _recurse_all_hypernyms(hypernym, all_hypernyms)

goal_synset = wn.synset('mammal.n.01')

dog_imgs = open('mammal_images.txt','w')
nodog_imgs = open('nomamm_images.txt','w')

with open('imagenet2012_val_synset_codes.txt','r') as f:
    for l in f:
        code = l.strip().split()[1]
        synset = wn.synset_from_pos_and_offset('n',int(code[1:]))
        hypernyms = []
        _recurse_all_hypernyms(synset, hypernyms)
        if goal_synset in hypernyms:
            dog_imgs.write(l.strip().split()[0]+'\n')
        else:
            nodog_imgs.write(l.strip().split()[0]+'\n')


