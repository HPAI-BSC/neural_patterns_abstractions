import numpy as np
import sys

sys.path.append('/gpfs/projects/bsc28/tiramisu_semantic_transfer/tiramisu_source/')
from tiramisu.tensorflow.core.backend import read_embeddings
import os
from collections import Counter


def find_layer_of_feature(feature):
    """
    This function find the layer correspondent to the given feature in a certain embedding.
    It returns the layer name of the feature.
    :param feature: a feature : int
    :param embedding: a dictionary with keys: ['embeddings', 'image_paths', 'image_labels', 'feature_scheme']
    :return:
    """
    # layers = embedding['feature_scheme'][()] ## Ordered dict.
    layers = {u'vgg16_model/relu4_2': (1664, 2176), u'vgg16_model/relu4_3': (2176, 2688),
              u'vgg16_model/relu4_1': (1152, 1664), u'vgg16_model/relu3_1': (384, 640),
              u'vgg16_model/relu3_3': (896, 1152), u'vgg16_model/relu3_2': (640, 896),
              u'vgg16_model/relu7': (8320, 12416), u'vgg16_model/relu6': (4224, 8320),
              u'vgg16_model/relu2_1': (128, 256), u'vgg16_model/relu2_2': (256, 384),
              u'vgg16_model/relu5_3': (3712, 4224), u'vgg16_model/relu5_2': (3200, 3712),
              u'vgg16_model/relu5_1': (2688, 3200), u'vgg16_model/relu1_2': (64, 128), u'vgg16_model/relu1_1': (0, 64)}
    layer = None
    for i in layers.values():
        if feature in range(i[0], i[1]):
            layer = i
            # print(feature, i)
    try:
        layer_name = list(layers.keys())[list(layers.values()).index(layer)]
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


def read_embedding(job_label, embedding_path,
                   data_path='../data/feature_lists', delete=False):
    """
    embedding_path expected:  returns /gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/1284246
    This function loads the embedding correspondent to the synset synset_name on the given embedding_path and extracts the
    features that have majority of images with value 1.
    :param synset_name:
    :param embedding_path:
    :return:
    """
    try:
        os.mkdir(data_path)
    except:
        pass
    try:
        synset_name = extract_synset_name(embedding_path)
    except:
        print('The synset of ', embedding_path, 'has no results file')
    print('-----------------')

    print('Going in with', job_label, synset_name)

    embedding_path = embedding_path + '/train/embeddings'
    try:
        (results, img_paths, labels, layers_dict) = read_embeddings(path_prefix=embedding_path)
    except:
        print('The embedding on ', embedding_path, ' Corresponding to: ', synset_name, 'is not working')
    # print(Counter(labels))
    dmp = np.copy(results == 1)

    # Get subset by synset
    pos_emb = dmp[[x == synset_name for x in labels]]
    neg_emb = dmp[[x != synset_name for x in labels]]
    if pos_emb.shape[0] == 0:
        print(synset_name, 'is broken')
        return
    print('Size of ' + synset_name + ' embedding', pos_emb.shape)
    print('Size of complementary embedding', neg_emb.shape)

    # Compute distribution per feature.
    pos_feature_counts = []
    neg_feature_counts = []
    pos_features_maj_ones = []
    neg_features_maj_ones = []
    counter = 0
    for p, n in zip(pos_emb.T, neg_emb.T):
        p_unique, p_counts = np.unique(p, return_counts=True)
        n_unique, n_counts = np.unique(n, return_counts=True)
        # If there is ony one value
        if len(p_counts) == 1:
            if p_unique[0] == True:
                pos_feature_counts.append(p_counts)
            else:
                pos_feature_counts.append(0)
        # if there are two values, get the second which corresponds to True
        else:
            if p_counts[1] > p_counts[0]:
                pos_features_maj_ones.append(counter)
            try:
                pos_feature_counts.append(p_counts[1])
            except IndexError:
                print('An error occurred when processing counts', p_counts)
        # If there is ony one value
        if len(n_counts) == 1:
            # its a full hit!
            if n_unique[0] == True:
                neg_feature_counts.append(n_counts)
            # Full miss
            else:
                neg_feature_counts.append(0)
        # if there are two values, get the second which corresponds to True
        else:
            if n_counts[1] > n_counts[0]:
                neg_features_maj_ones.append(counter)
            try:
                neg_feature_counts.append(n_counts[1])
            except IndexError:
                print('An error occurred when processing counts', n_counts)

        counter += 1

    path_pos_maj_ones = data_path + synset_name + '_pos_features_maj_ones.npz'
    path_neg_maj_ones = data_path + synset_name + '_neg_features_maj_ones.npz'

    np.savez(path_pos_maj_ones, pos_features=pos_features_maj_ones, layers=layers_dict)
    np.savez(path_neg_maj_ones, pos_features=neg_features_maj_ones, layers=layers_dict)

    ####################
    # Delete the embedding once we have the data:
    if delete:
        import shutil
        shutil.rmtree(embedding_path)


def main():
    location = '/gpfs/projects/bsc28/tiramisu_semantic_transfer/embeddings/'
    folders = next(os.walk(location))[1]
    print(folders)
    for job_label in folders:
        _path = location + job_label
        read_embedding(job_label, _path)


if __name__ == "__main__":
    from time import time

    ini = time()
    main()
    print(time() - ini)
