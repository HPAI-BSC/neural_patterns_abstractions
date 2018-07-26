import sys
sys.path.append('/gpfs/projects/bsc28/tiramisu_semantic_transfer/')
import numpy as np
from tiramisu.tensorflow.core.backend import read_embeddings

embeddings = ['1284246']
labels = ['dog']

for emb,lab in zip(embeddings,labels):
    (results, img_paths, labels, layers_dict) = read_embeddings(path_prefix="/gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/1284246/train/embeddings_0")
    print '-----------------'
    print 'Going in with',emb,' ',lab
    dmp = np.copy(results == 1)
    #Currently exploring only the positives
    #dmn = np.copy(results == -1)
    #dm = np.concatenate((dmp , dmn), axis=1)

    #Get subset by synset
    pos_emb = dmp[[x==lab for x in labels]]
    neg_emb = dmp[[x!=lab for x in labels]]
    print 'Size of '+lab+' embedding',pos_emb.shape
    print 'Size of complementary embedding',neg_emb.shape

    #Compute distribution per feature.
    pos_feature_counts = []
    neg_feature_counts = []
    pos_full_hits = 0
    pos_full_misses = 0
    neg_full_hits = 0
    neg_full_misses = 0
    pos_features_maj_ones = []
    neg_features_maj_ones = []
    counter = 0
    for p,n in zip(pos_emb.T,neg_emb.T):
        p_unique, p_counts = np.unique(p, return_counts=True)
        n_unique, n_counts = np.unique(n, return_counts=True)
        #If there is ony one value
        if len(p_counts)==1:
            #its a full hit!
            if p_unique[0] == True:
                pos_full_hits+=1
                pos_feature_counts.append(p_counts)
            #Full miss
            else:
                pos_full_misses+=1
                pos_feature_counts.append(0)
        #if there are two values, get the second which corresponds to True
        else:
            if p_counts[1]>p_counts[0]:
                pos_features_maj_ones.append(counter)
            try:
                pos_feature_counts.append(p_counts[1])
            except IndexError:
                print 'An error occurred when processing counts',p_counts
        #If there is ony one value
        if len(n_counts)==1:
            #its a full hit!
            if n_unique[0] == True:
                neg_full_hits+=1
                neg_feature_counts.append(n_counts)
            #Full miss
            else:
                neg_full_misses+=1
                neg_feature_counts.append(0)
        #if there are two values, get the second which corresponds to True
        else:
            if n_counts[1]>n_counts[0]:
                neg_features_maj_ones.append(counter)
            try:
                neg_feature_counts.append(n_counts[1])
            except IndexError:
            print 'An error occurred when processing counts',n_counts
        counter+=1
    print lab+' Positive full hits:',pos_full_hits, ' full misses:',pos_full_misses
    print lab+' Negative full hits:',neg_full_hits, ' and full misses:',neg_full_misses
    #print 'Features with majority of ones:',features_maj_ones

    #Store
    #np.save(open('liv_embedding_feature_counts.npy','w'),feature_counts)
    #Plot
    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.mlab as mlab
    import matplotlib.pyplot as plt
    #Get only the positive cases. its simetrical to 50K
    n, bins, patches = plt.hist(pos_feature_counts, 500, facecolor='green', alpha=0.75)
    plt.axvline(pos_emb.shape[0]/2, color='k', linestyle='solid')
    plt.savefig('positive_features_distributions_'+lab+'.pdf')
    plt.clf()
    n, bins, patches = plt.hist(neg_feature_counts, 500, facecolor='green', alpha=0.75)
    plt.axvline(neg_emb.shape[0]/2, color='k', linestyle='solid')
    plt.savefig('negative_features_distributions_'+lab+'.pdf')
    plt.clf()

    pos_f = open(lab+'_pos_features_maj_ones.txt','w')
    for item in pos_features_maj_ones:
      pos_f.write("%s\n" % item)
    pos_f.close()
    neg_f = open(lab+'_neg_features_maj_ones.txt','w')
    for item in neg_features_maj_ones:
      neg_f.write("%s\n" % item)
    neg_f.close()


###Load most frequent feature values separately
#most_freq_vals_dead = []
#most_freq_vals_live = []
#for d,a in zip(dead_emb.T,live_emb.T):
#    most_freq_vals_dead.append(np.argmax(np.bincount(d)))
#    most_freq_vals_live.append(np.argmax(np.bincount(a)))
#print 'For dog hyponyms, the number of features with more frequent 1s are:',np.bincount(most_freq_vals_dead)[1]
#print 'For nodog hyponyms, the number of features with more frequent 1s are:',np.bincount(most_freq_vals_live)[1]
#
##Get indices of features with 1
#dead_indices = np.argwhere(most_freq_vals_dead)
#live_indices = np.argwhere(most_freq_vals_live)
#print live_indices
##Get the count by layer
#dead_layer_counts = {}
#live_layer_counts = {}
#for d in dead_indices:
#    for k,v in layers_dict.iteritems():
#        if v[0] < d < v[1]:
#            if k in dead_layer_counts.keys():
#                dead_layer_counts[k]+=1
#            else:
#                dead_layer_counts[k]=1
#for l in live_indices:
#    for k,v in layers_dict.iteritems():
#        if v[0] < l < v[1]:
#            if k in live_layer_counts.keys():
#                live_layer_counts[k]+=1
#            else:
#                live_layer_counts[k]=1
#print live_layer_counts


#NEXT: sort layers. find subsets for hyponym embedding extraction (vs mammal?). look for relations between features of same synset (weights between layers?)
