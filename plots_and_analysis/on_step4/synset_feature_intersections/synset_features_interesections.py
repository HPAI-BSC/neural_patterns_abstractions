from __future__ import division
import os
import numpy as np
import itertools

def main():
    #Path to data
    step4_path = '/gpfs/projects/bsc28/semantic_transfer_scripts/data'
    #Initialize data structure
    pos_feats = {}
    #Load data
    for f in os.listdir(step4_path):
        #Skip negative ones
        if 'neg' in f:
            continue
        #Read the data
        x = np.load(os.path.join(step4_path,f))['pos_features']
        name = f.split('pos')[0][:-1]
        pos_feats[name] = x
    #Compute intersections
    synsets = pos_feats.keys()
    intersects = []
    for pair in itertools.combinations(synsets,2):
        try:
            first_val = len(set(pos_feats[pair[0]]).intersection(set(pos_feats[pair[1]])))/len(pos_feats[pair[0]])
        except ZeroDivisionError:
            first_val = 0.0
        try:
            second_val = len(set(pos_feats[pair[0]]).intersection(set(pos_feats[pair[1]])))/len(pos_feats[pair[1]])
        except ZeroDivisionError:
            second_val = 0.0
        intersects.append(((pair[0],pair[1]),first_val))
        intersects.append(((pair[1],pair[0]),second_val))
       
    #Sort data structure
    sorted_intersects = sorted(intersects, key=lambda x: x[1], reverse=True)
    with open('data/synset_feature_interesections.txt','w') as f:
        for x in sorted_intersects:
            f.write(x[0][0]+' '+x[0][1]+' '+str(x[1])+'\n')

"""
This code generates the amount of synset feature interesection for every pair of synsets, and then normalizes by the size of each synset feature size (2 different normalizations per pair). Results are stored in '../plots/synset_feature_interesections.txt' "
"""

if __name__ == "__main__":
        from time import time
        ini = time()
        main()
        print(time() - ini)

