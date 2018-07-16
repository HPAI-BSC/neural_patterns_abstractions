from __future__ import division
import numpy as np
import sys
sys.path.append('/gpfs/projects/bsc28/tiramisu_semantic_transfer/tiramisu_source/')
from tiramisu.tensorflow.core.backend import read_embeddings
import os
from collections import Counter, OrderedDict
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

layers = OrderedDict([(u'vgg16_model/relu1_1', (0, 64)), (u'vgg16_model/relu1_2', (64, 128)), (u'vgg16_model/relu2_1', (128, 256)), (u'vgg16_model/relu2_2', (256, 384)), (u'vgg16_model/relu3_1', (384, 640)), (u'vgg16_model/relu3_2', (640, 896)), (u'vgg16_model/relu3_3', (896, 1152)), (u'vgg16_model/relu4_1', (1152, 1664)), (u'vgg16_model/relu4_2', (1664, 2176)), (u'vgg16_model/relu4_3', (2176, 2688)), (u'vgg16_model/relu5_1', (2688, 3200)), (u'vgg16_model/relu5_2', (3200, 3712)), (u'vgg16_model/relu5_3', (3712, 4224)), (u'vgg16_model/relu6', (4224, 8320)), (u'vgg16_model/relu7', (8320, 12416))])

def find_layer_of_feature(feature, embedding):
        """
        This function find the layer correspondent to the given feature in a certain embedding.
        It returns the layer name of the feature.
        :param feature: a feature : int
        :param embedding: a dictionary with keys: ['embeddings', 'image_paths', 'image_labels', 'feature_scheme']
        :return:
        """
        layers = embedding['feature_scheme'][()] ## Ordered dict.
        layer = None
        for i in layers.values():
                if feature in range(i[0], i[1]):
                        layer = i
                        # print(feature, i)
        try:
                layer_name =  list(layers.keys())[list(layers.values()).index(layer)]
                return layer_name
        except:
                print('Feature not in range')
        return None

def extract_synset_name(path):
        """
        /gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/1284246
        :param path:
        :return:
        """
        _path = path + '/results.txt'
        synset_name = ''
        with open(_path, 'r') as f:
                for l in f:
                        if 'dataset' in l:
                                synset_name = l.split(' ')[1].strip()
        return str(synset_name)


def read_embedding( job_label,embedding_path,data_path='/gpfs/projects/bsc28/semantic_transfer_scripts/data/', delete=False):
        """
        embedding_path expected:  returns /gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/1284246
        :param synset_name:
        :param embedding_path:
        :return:
        """
        synset_name = extract_synset_name(embedding_path)
        print('Going in with', job_label, synset_name)
        embedding_path = embedding_path + '/train/embeddings'
        (results, img_paths, labels, layers_dict) = read_embeddings(path_prefix=embedding_path)
        dmp = np.copy(results == 1)

        # Get subset by synset
        pos_emb = dmp[[x == synset_name for x in labels]]
        #neg_emb = dmp[[x != synset_name for x in labels]]
        if pos_emb.shape[0] == 0:
                print(synset_name, 'is broken')
                return
        print('Size of ' + synset_name + ' embedding', pos_emb.shape)
        #print('Size of complementary embedding', neg_emb.shape)

        # Compute distribution per feature.
        pos_feature_counts = []
        #neg_feature_counts = []
        counter = 0
        for p in pos_emb.T:
        #for p, n in zip(pos_emb.T, neg_emb.T):
                p_unique, p_counts = np.unique(p, return_counts=True)
                #n_unique, n_counts = np.unique(n, return_counts=True)
                # If there is only one value
                if len(p_counts) == 1:
                        if p_unique[0] == True:
                                pos_feature_counts.append(p_counts)
                        else:
                                pos_feature_counts.append(0)
                # if there are two values, get the second which corresponds to True
                else:
                        try:
                                pos_feature_counts.append(p_counts[1])
                        except IndexError:
                                print('An error occurred when processing counts', p_counts)
                # If there is ony one value
                #if len(n_counts) == 1:
                #        # its a full hit!
                #        if n_unique[0] == True:
                #                neg_feature_counts.append(n_counts)
                #        # Full miss
                #        else:
                #                neg_feature_counts.append(0)
                ## if there are two values, get the second which corresponds to True
                #else:
                #        if n_counts[1] > n_counts[0]:
                #                neg_features_maj_ones.append(counter)
                #        try:
                #                neg_feature_counts.append(n_counts[1])
                #        except IndexError:
                #                print('An error occurred when processing counts', n_counts)

                counter += 1
        return pos_feature_counts, pos_emb.shape[0]

def plot_synset_layer_distribution(pos_feature_counts, size, name):
    """
    Plot histograms of layer distribution. Stores 2 figures (one normalized, the other one
    with absolute values) in "../plots/" as pdfs.
    :pos_feature_counts: list of counts of length equal to the number of features. Each value
                         indicates the number of images within the synset with a 1 for that feature.
    :size: number of images within the synset
    :name: name of the synset
    """
    #which thresholds are going to be considered?
    thresholds = [30,40,50,60,70,80,90,100]
    #Initialize the data structures for storing the plot points
    data_points_by_layer = np.zeros((len(layers.keys()),len(thresholds)))
    data_points_by_layer_norm = np.zeros((len(layers.keys()),len(thresholds)))
    #keep track of layer number
    counter = 0
    #Generate data per layer
    for l in layers.keys():
        layer_range = layers[l]
        #Generate data per threshold level
        for idx_t,t in enumerate(thresholds):
            #Get number of features within layer and above threshold
            val = len([x for idx,x in enumerate(pos_feature_counts) if idx >= layer_range[0] and idx <= layer_range[1] and x/size > (t*0.01)])
            #Store raw data, store normalized data
            data_points_by_layer[counter][idx_t] = val
            val = val/(layer_range[1]-layer_range[0])
            data_points_by_layer_norm[counter][idx_t] = val
        counter+=1
    #plot histogram
    palette = plt.get_cmap('plasma')
    types = ['o','*','X']
    counter = 0
    for d,l in zip(data_points_by_layer,layers.keys()):
        plt.plot(thresholds,d, '-'+types[counter%3], label=l, color=palette(counter))
        counter+=20
    plt.legend(loc='upper right')
    plt.savefig('../plots/layer_distribution_pos_feats_'+name+'.pdf')
    #plot histogram normalized
    plt.cla()
    palette = plt.get_cmap('plasma')
    types = ['o','*','X']
    counter = 0
    for d,l in zip(data_points_by_layer_norm,layers.keys()):
        plt.plot(thresholds,d, '-'+types[counter%3], label=l, color=palette(counter))
        counter+=20
    plt.legend(loc='upper right')
    plt.savefig('../plots/layer_distribution_pos_feats_norm_'+name+'.pdf')
    

"""
This code generates plots showing the distribution of "synset features" through layers. For each synset two plots are generated. Each plot shows the distribution of features per layer, as the threshold of minimum activated images varies. One plot shows the data in absolute number of features, and the other normalizing by the number of features in the layer. Plots are stored in "../plots"
REQUIRES: 
    -Access to the results of step 3 (var location).
    -Access to tiramisu methods (see top of the file)
    -The embeddings to be from vgg16 (see top of the file, var layers)
"""
def main():
        location =  '/gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/'
        folders =  next(os.walk(location))[1]
        for job_label in folders[2:]:
                _path = location + job_label
                try:
                    name = extract_synset_name(_path)
                    pos_feature_counts, size = read_embedding(job_label,_path)
                    plot_synset_layer_distribution(pos_feature_counts, size, name)
                except IOError:
                    #Likely cause in extract_synset_name (missing results.txt file)
                    print 'WARNING1: SKIPPING CORRUPTED EMBEDDING',_path
                except TypeError:
                    #Likely cause in read_embedding (empty embedding)
                    print 'WARNING2: SKIPPING CORRUPTED EMBEDDING',_path


if __name__ == "__main__":
        from time import time
        ini = time()
        main()
        print(time() - ini)
