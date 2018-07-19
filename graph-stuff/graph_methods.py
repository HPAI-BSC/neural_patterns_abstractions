"""
In this file i will put the basic methods to represent the embedding graph.
"""
import numpy as np
from collections import OrderedDict
from matplotlib import pyplot as plt
from time import time
from datetime import timedelta

WEIGHTS_PATH = '../data/models/vgg16_imagenet_weights.npy'
ARRAY_PATH = '../data/arrays_maj_ones/'
PLOT_PATH = '../data/plots/'

conv_layers = OrderedDict(sorted({'conv1_1': (0, 64), 'conv1_2': (64, 128),
                                  'conv2_1': (128, 256), 'conv2_2': (256, 384),
                                  'conv3_1': (384, 640), 'conv3_2': (640, 896), 'conv3_3': (896, 1152),
                                  'conv4_1': (1152, 1664), 'conv4_2': (1664, 2176), 'conv4_3': (2176, 2688),
                                  'conv5_1': (2688, 3200), 'conv5_2': (3200, 3712), 'conv5_3': (3712, 4224),
                                  'fc6': (4224, 8320), 'fc7': (8320, 12416)
                                  }.items(), key=lambda t: t[1]))

relu_layers = {u'vgg16_model/relu4_2': (1664, 2176), u'vgg16_model/relu4_3': (2176, 2688),
               u'vgg16_model/relu4_1': (1152, 1664), u'vgg16_model/relu3_1': (384, 640),
               u'vgg16_model/relu3_3': (896, 1152), u'vgg16_model/relu3_2': (640, 896),
               u'vgg16_model/relu7': (8320, 12416), u'vgg16_model/relu6': (4224, 8320),
               u'vgg16_model/relu2_1': (128, 256), u'vgg16_model/relu2_2': (256, 384),
               u'vgg16_model/relu5_3': (3712, 4224), u'vgg16_model/relu5_2': (3200, 3712),
               u'vgg16_model/relu5_1': (2688, 3200), u'vgg16_model/relu1_2': (64, 128), u'vgg16_model/relu1_1': (0, 64)}


def find_layer_of_feature(feature, layer_notation='conv'):
    """
    This function find the layer correspondent to the given feature in a certain embedding.
    It returns the layer name of the feature.

    There are two different notations:
    - The used by the FNE: vgg16_model/relux_y
    - The used by the vgg16 weights: 'convx_y

    :param feature: a feature : int
    :param embedding: a dictionary with keys: ['embeddings', 'image_paths', 'image_labels', 'feature_scheme']
    :return:
    """
    # layers = embedding['feature_scheme'][()] ## Ordered dict.
    if layer_notation == 'relu':
        layers = relu_layers

    else:
        layers = conv_layers

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


def separate_per_layer(feature_array):
    """
    It returns a dicctionary sepparatin on the layers the features of the feature array
    :param feature_array:
    :return:
    """
    feature_dict = {}
    for f in feature_array:
        layer_name = find_layer_of_feature(f)
        try:
            feature_dict[layer_name] = np.append(feature_dict[layer_name], f)
        except:
            feature_dict[layer_name] = np.array([f])
    # print(feature_dict)
    return OrderedDict(sorted(feature_dict.items(), key=lambda t: t[0]))



def load_weights(path):
    all_weights = np.load(path, encoding='latin1')[()]
    # I take the weights without bias
    weights = {key: all_weights[key][0] for key in all_weights.keys()}
    return weights


def plot_weights():
    weights = load_weights(WEIGHTS_PATH)
    for k in weights.keys():
        print(k, weights[k].min(), weights[k].max())
        plt.figure()
        plt.hist(weights[k].flatten())
        plt.title(k)
        plt.savefig(PLOT_PATH + k + '.png')


def fne_feature_to_vgg16_block(feature, all_weights=None):
    """
    This function returns the block of weights correspondent to each feature of the FNE.
    :param feature:
    :return:
    """
    layer_name = find_layer_of_feature(feature)
    layer = conv_layers[layer_name]
    if not all_weights:
        weights_of_layer = load_weights(WEIGHTS_PATH)[layer_name]
    else:
        weights_of_layer = all_weights[layer_name]
    f = feature - layer[0]
    if 'conv' in layer_name:
        weights = weights_of_layer[:, :, :, f]
    else:
        weights = weights_of_layer[:, f]
    return weights


# def vgg16_feature_to_target_feature_fne(feature_coord, layer):
#     """
#     This function returns the FNE feature correspondent to the coordinates on one layer.
#     The layer can be conv or fc.
#     The feature coord can be:
#     if conv:
#         feature_coord = [x,y,z,k]
#     if fc:
#         feature_coord = [x,y]
#
#     :param feature:
#     :return:
#     """
#     layer_before = conv_layers.items()[conv_layers.keys().index(layer) - 1]
#     if 'conv' in layer:
#         z = feature_coord[3]
#         feature_destiny = feature_coord[3] + conv_layers[layer][0]
#         feature_origin = 0
#
#     else:
#
#         feature_destiny = feature_coord[1] + conv_layers[layer][0]
#
#     feature = [feature_origin, feature_destiny]
#     return feature


def weigh_distribution(features_array, name='patata'):
    all_weights = load_weights(WEIGHTS_PATH)

    conv_features = [k for k in features_array if k < 4224]
    weights_conv = np.array([])
    weights_conv_mean = np.array([])
    for f in conv_features:
        block = fne_feature_to_vgg16_block(f, all_weights)
        weights_conv = np.append(weights_conv, block.flatten())
        weights_conv_mean = np.append(weights_conv_mean, np.mean(block.flatten()))

    if len(weights_conv) > 0:
        plt.figure()
        plt.hist(weights_conv, bins=10, range=(-.05, 0.05))
        plt.savefig(PLOT_PATH + name + '_conv.png')

    # plt.figure()
    # plt.hist(weights_conv_mean, bins=10,range=(-.02,0.02))
    # plt.savefig(PLOT_PATH + name + '_conv_mean.png')

    # plt.show()

    fc_features = [k for k in features_array if k >= 4224]
    weights_fc = np.array([])

    for f in fc_features:
        weights_fc = np.append(weights_fc, fne_feature_to_vgg16_block(f, all_weights).flatten())
    if len(weights_fc) > 0:
        plt.figure()
        plt.hist(weights_fc, bins=10)  # , range=(-.05, 0.05))
        plt.savefig(PLOT_PATH + name + '_fc.png')


def extract_weights_of_interest(all_interest_nodes):
    """
    The main idea of this code is extract the connexion between neurons with income connection such that have value one on
    their corresponding fne value is one.
    :param all_interest_nodes:
    :return:
    """
    weights = np.array([])
    weights_conv = np.array([])
    weights_fc = np.array([])
    layers = list(all_interest_nodes.keys())
    all_weights = load_weights(WEIGHTS_PATH)

    per_layer = {}

    for l in range(len(layers)):
        layer = layers[l]
        if layer == 'fc7':
            break
        next_layer = layers[l+1]
        local_weights = []
        if 'conv' in next_layer:
            for feature2 in all_interest_nodes[next_layer]:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in all_interest_nodes[layer]:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_conv = np.append(weights_conv, w[:, :, feature1 - conv_layers[layer][0]].flatten())
        if next_layer == 'fc6':
            for feature2 in all_interest_nodes[next_layer]:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                wa = np.reshape(w, (7, 7, 512))
                for feature1 in all_interest_nodes[layer]:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, wa[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_fc = np.append(weights_fc, wa[:, :, feature1 - conv_layers[layer][0]].flatten())
        if next_layer == 'fc7':
            for feature2 in all_interest_nodes[next_layer]:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in all_interest_nodes[layer]:
                    local_weights = np.append(local_weights, w[feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[feature1 - conv_layers[layer][0]])
                    weights_fc = np.append(weights_fc, w[feature1 - conv_layers[layer][0]])
        per_layer[layer] = local_weights
    return weights, weights_conv, weights_fc, per_layer


def plot_hist(a, name):
    plt.figure()
    plt.hist(a, bins=10)
    plt.savefig(PLOT_PATH + name + '.png')


def plot_weights_of_interest(all_interest_nodes, name):
    # w, w_c, w_fc = extract_weights_of_interest(all_interest_nodes)
    # plot_hist(w, 'all_interest_weights_' + name)
    # plot_hist(w_c, 'conv_interest_weights_' + name)
    # plot_hist(w_fc, 'fc_interest_weights_' + name)
    # print('interest weights extracted and ploted')
    # w, w_c, w_fc = extract_weights_of_interest_with_normal_origin(all_interest_nodes)
    # print('calculated')
    # plot_hist(w, 'all_interest_weights_mixed_origin_' + name)
    # plot_hist(w_c, 'conv_interest_weights_mixed_origin_' + name)
    # plot_hist(w_fc, 'fc_interest_weights_mixed_origin_' + name)
    # print('interest weights origin extracted and ploted')

    w, w_c, w_fc = extract_weights_of_interest_with_normal_destination(all_interest_nodes)
    plot_hist(w, 'all_interest_weights_mixed_destination' + name)
    plot_hist(w_c, 'conv_interest_weights_mixed_destination' + name)
    plot_hist(w_fc, 'fc_interest_weights_mixed_destination' + name)

    print('interest weights destiny extracted and ploted')


def extract_weights_of_interest_with_normal_origin(all_interest_nodes):
    """
    The main idea of this code is extract the connexion between neurons with income connection such that have value one on
    their corresponding fne value is one.
    :param feture_dict:
    :return:
    """
    weights = np.array([])
    weights_conv = np.array([])
    weights_fc = np.array([])
    layers = list(all_interest_nodes.keys())
    all_weights = load_weights(WEIGHTS_PATH)

    per_layer = {}

    for l in range(len(layers)):
        layer = layers[l]

        if layer == 'fc7':
            break

        local_weights = []

        next_layer = layers[l + 1]

        all_nodes_origin = np.array(range(conv_layers[layer][0], conv_layers[layer][1]))
        all_nodes_destination = np.array(range(conv_layers[next_layer][0], conv_layers[next_layer][1]))

        interest_nodes_origin = all_interest_nodes[layer]
        interest_nodes_destination = all_interest_nodes[next_layer]
        normal_nodes_origin = np.setdiff1d(all_nodes_origin, interest_nodes_origin)
        normal_nodes_destination = np.setdiff1d(all_nodes_destination, interest_nodes_destination)


        interest_nodes = interest_nodes_destination
        normal_nodes = normal_nodes_origin

        if 'conv' in next_layer:
            for feature2 in interest_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in normal_nodes:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_conv = np.append(weights_conv, w[:, :, feature1 - conv_layers[layer][0]].flatten())

        if next_layer == 'fc6':
            for feature2 in interest_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                wa = np.reshape(w, (7, 7, 512))
                for feature1 in normal_nodes:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, wa[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_fc = np.append(weights_fc, wa[:, :, feature1 - conv_layers[layer][0]].flatten())

        if next_layer == 'fc7':
            for feature2 in interest_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in normal_nodes:
                    local_weights = np.append(local_weights, w[feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[feature1 - conv_layers[layer][0]])
                    weights_fc = np.append(weights_fc, w[feature1 - conv_layers[layer][0]])

        per_layer[layer] = local_weights
    return weights, weights_conv, weights_fc, per_layer


def extract_weights_of_interest_with_normal_destination(all_interest_nodes):
    """
    The main idea of this code is extract the connexion between neurons with income connection such that have value one on
    their corresponding fne value is one.
    :param feture_dict:
    :return:
    """
    weights = np.array([])
    weights_conv = np.array([])
    weights_fc = np.array([])
    layers = list(all_interest_nodes.keys())
    all_weights = load_weights(WEIGHTS_PATH)

    per_layer = {}

    for l in range(len(layers)):
        layer = layers[l]
        if layer == 'fc7':
            break

        next_layer = layers[l + 1]

        all_nodes_origin = np.array(range(conv_layers[layer][0], conv_layers[layer][1]))
        all_nodes_destination = np.array(range(conv_layers[next_layer][0], conv_layers[next_layer][1]))

        interest_nodes_origin = all_interest_nodes[layer]
        interest_nodes_destination = all_interest_nodes[next_layer]
        normal_nodes_origin = np.setdiff1d(all_nodes_origin, interest_nodes_origin)
        normal_nodes_destination = np.setdiff1d(all_nodes_destination, interest_nodes_destination)

        interest_nodes = interest_nodes_origin
        normal_nodes = normal_nodes_destination

        local_weights = []
        if 'conv' in next_layer:
            for feature2 in normal_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in interest_nodes:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_conv = np.append(weights_conv, w[:, :, feature1 - conv_layers[layer][0]].flatten())

        if next_layer == 'fc6':
            for feature2 in normal_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                wa = np.reshape(w, (7, 7, 512))
                for feature1 in interest_nodes:
                    local_weights = np.append(local_weights, w[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, wa[:, :, feature1 - conv_layers[layer][0]].flatten())
                    weights_fc = np.append(weights_fc, wa[:, :, feature1 - conv_layers[layer][0]].flatten())

        if next_layer == 'fc7':
            for feature2 in normal_nodes:
                w = fne_feature_to_vgg16_block(feature2, all_weights)
                for feature1 in interest_nodes:
                    local_weights = np.append(local_weights, w[feature1 - conv_layers[layer][0]].flatten())
                    weights = np.append(weights, w[feature1 - conv_layers[layer][0]])
                    weights_fc = np.append(weights_fc, w[feature1 - conv_layers[layer][0]])
        per_layer[layer] = local_weights

    return weights, weights_conv, weights_fc, per_layer


def main():
    hunting_dog = np.load(ARRAY_PATH + 'hunting_dog_pos_features_maj_ones.npz')['pos_features']

    ss = hunting_dog

    ss_dictionary = separate_per_layer(ss)

    plot_weights_of_interest(ss_dictionary, 'hunting_dog')
    # for k in ss_dictionary.keys():
    #     weigh_distribution(ss_dictionary[k], 'hunting_dog_' + k)


# plt.figure(1)
# sns.distplot(ss)
# plt.show()
# weigh_distribution(ss)


if __name__ == "__main__":
    init = time()
    main()
    print('time:', timedelta(seconds=time() - init))
