"""
This code reads a file relating images with synsets, of the following form:

    ILSVRC2012_val_00000001.JPEG n01751748
    ILSVRC2012_val_00000002.JPEG n09193705
    ILSVRC2012_val_00000003.JPEG n02105855
    ...

Where the first column correspond to image names, and the second to the wordnet synset of the corresponding image label.
The code identifies all synsets in WordNet (notice this may include synsets not found in the file) which are composed by
 a given range of images (both min. and max.). Inheritance is assumed, such that images labeled to synset "dog" are
 counted for synset "mammal", "living_thing", etc.
A file is stored for each synset satisfying the restrictions, containing the images that compose it.

For faster calculations the code generates the following temporary files:
        all_ss : The set of synsets that appear in the dataset.
        all_hypers: The set of hypernyms of all_ss
        interest_ss: The set of synsets that have more a number of images within the range, including hyponym images.
        imagenet.....npz: The information of the dataset separated on two numpy array.
Also it will provide with a list of the candidate synsets to delete. Based on the synsets that have the same number of
 images and are hyponyms/hypernyms between them.
"""

from nltk.corpus import wordnet as wn
import numpy as np
from datetime import timedelta
import time
import functools as fu
import os

SYNSET_PARTITION_PATH = '../data/synset_partitions/'

try:
    os.mkdir(SYNSET_PARTITION_PATH)
except:
    pass

NO_SYNSET_PARTITION_PATH = SYNSET_PARTITION_PATH + '/no_synset/'

try:
    os.mkdir(NO_SYNSET_PARTITION_PATH)
except:
    pass


def ss_to_text(synset):
    """ Returns the name of a given synset"""
    return str(synset)[8:-7]


def get_wn_ss(imagenet_id):
    """
    Transforms an imagenet id into a wordnet synset
    :param imagenet_id:
    :return:
    """
    return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])


def img_ids_to_text(samples):
    """
    Transforms an array of imagenet ids to an array of its names
    :param samples:
    :return:
    """
    ss = [ss_to_text(get_wn_ss(k)) for k in samples]
    return ss


def get_in_id(wordnet_ss):
    """
    Transforms a worndet synset into a imagenet id
    Input: Synset
    :param wordnet_ss:
    :return: imagenet id (string)
    """
    wn_id = wn.ss2of(wordnet_ss)
    return wn_id[-1] + wn_id[:8]


def obtain_hypernyms(code):
    """
    This function returns a list of the hypernims of the imagenet id code.
    :param code:imagenet id
    :return:
    """
    ss = wn.synset_from_pos_and_offset('n', int(code[1:]))
    hyper = lambda s: s.hypernyms()
    return np.array(list(get_in_id(k) for k in ss.closure(hyper)))


def obtain_hyponims(code):
    """
    This function returns a list of the hyponims of the imagenet id code.
    :param code:imagenet id
    :return:
    """
    ss = wn.synset_from_pos_and_offset('n', int(code[1:]))
    hypo = lambda s: s.hyponyms()
    return np.array(list(get_in_id(k) for k in ss.closure(hypo)))


def generate_all_ss(image_synset_file_path):
    """
    Generates a file with all the diferent synsets that appear in image_synset_file_path without repetitions.
    :return:
    """
    all_ss = np.array([])
    with open(image_synset_file_path, 'r') as f:
        for l in f:
            code = l.strip().split()[1]
            if np.isin(code, all_ss).sum():
                pass
            else:
                all_ss = np.append(all_ss, code)

    np.savez('../data/all_ss.npz', ss=all_ss)
    return all_ss


def generate_all_hypers(image_synset_file_path):
    """
    Generates a file with all the hypernyms of the set of synsets that appear in image_synset_file_path without repetitions.
    :return:
    """
    try:
        all_ss = np.load('../data/all_ss.npz')['ss']
    except:
        all_ss = generate_all_ss(image_synset_file_path)

    all_hypers = np.array([])
    for ss in all_ss:
        all_hypers = np.append(all_hypers, ss)
        hypers = obtain_hypernyms(ss)
        for h in hypers:
            if np.isin(h, all_hypers).sum():
                pass
            else:
                all_hypers = np.append(all_hypers, h)

    print('Total hypernyms found:',len(all_hypers))
    np.savez('../data/all_hypers.npz', ss=all_hypers)
    return all_hypers


def generate_all_codes_and_images_as_array(image_synset_file_path):
    """
    Creates a npz with information from image_synset_file_path.
    It is divided into two arrays:
    - all_imgs: the names of the images of the original file.
    - all_ss: the imagenet id of the class of each image.
    :return:
    """
    all_ss = np.array([])
    all_imgs = np.array([])
    with open(image_synset_file_path, 'r') as f:
        for l in f:
            code = l.strip().split()[1]
            img = l.strip().split()[0]
            all_ss = np.append(all_ss, code)
            all_imgs = np.append(all_imgs, img)

    np.savez('../data/imagenet2012_val_synset_codes_as_array.npz', ss=all_ss, imgs=all_imgs)
    return all_ss, all_imgs


def index_and_hyponims_from_label(ss, image_synset_file_path):
    """
    This function generates an index of the position of the synset or its hyponyms in the format:
    index[x] = 1 if the synset ss or one of its hyponyms is in position x.
    index[0] = 0 else.
    :param ss:
    :return:
    """
    try:
        all_labels = np.load('../data/imagenet2012_val_synset_codes_as_array.npz')['ss']
    except:
        all_labels = generate_all_codes_and_images_as_array(image_synset_file_path)[0]
    try:
        dif_labels = np.load('../data/all_ss.npz')['ss']
    except:
        dif_labels = generate_all_ss(image_synset_file_path)
    hyp = np.array(obtain_hyponims(ss))
    hyp_in_labels = np.isin(hyp, dif_labels)
    hyp = hyp[hyp_in_labels]
    hyp = np.append(ss, hyp)
    indexs = [k == all_labels for k in hyp]
    index = fu.reduce(lambda a, b: a + b, indexs)
    return index


def generate_partition(goal_synset, all_ss, image_synset_file_path, min_synset_freq, max_synset_freq, format='npz'):
    """
    This function makes the partition of the images with the criteria if is goal_synset (or hyponym of it) or not.
    Writes two files with this partion in ../data/goal_synset/ if the min_synset_freq and max_synset_freq are satisfied.
    """
    ss = np.array([])
    no_ss = np.array([])
    hyper_list = []

    with open(image_synset_file_path, 'r') as f:
        for l in f:
            code = l.strip().split()[1]
            hypernyms = obtain_hypernyms(code)

            if goal_synset in hypernyms:
                if code in all_ss:
                    hyper_list = np.append(hyper_list, str(wn.synset_from_pos_and_offset('n', int(code[1:]))))
                ss = np.append(ss, l.strip().split()[0])
            else:
                no_ss = np.append(no_ss, l.strip().split()[0])

    if len(ss) >= min_synset_freq and len(ss) <= max_synset_freq:
        np.savetxt(SYNSET_PARTITION_PATH + img_ids_to_text([goal_synset])[0] + '_hypernyms.txt',
                   hyper_list, fmt="%s")
        np.savez(SYNSET_PARTITION_PATH + img_ids_to_text([goal_synset])[0] + '_images.npz',
                 name=str(goal_synset), code=goal_synset, imgs=ss, hyper=hyper_list)
        # np.savez('NO_SYNSET_PARTITION_PATH + img_ids_to_text([goal_synset])[0] + '_images.npz', name=str(goal_synset), code=goal_synset, imgs=no_ss)
        if format == 'txt':
            np.savetxt(SYNSET_PARTITION_PATH + img_ids_to_text([goal_synset])[0] + '_images.txt', ss,
                       fmt="%s")
            np.savetxt(NO_SYNSET_PARTITION_PATH + 'no_' + img_ids_to_text([goal_synset])[0] + '_images.txt',
                       no_ss, fmt="%s")

        return 1
    return 0


def delete_repeated_ss(number_of_images):
    """
    This funciton selects the repeated synsets and returns them as an array.
    To do so it uses a dictionary with the number of images as a key and a list of the synsets with that number of
    images as a values.
    It selects to synsets to delete the ones with the same number of images
    :param number_of_images: dict[int] = list() Is a dictionary
    :return:
    """
    to_delete = list()
    with open('../data/synsets_with_same_number_of_images.txt','w') as f:
        for number in number_of_images.keys():
            if len(number_of_images[number]) == 2:
                if np.isin(get_wn_ss(number_of_images[number][0]), get_wn_ss(number_of_images[number][1]).hyponyms()):
                    to_delete.append(number_of_images[number][1])
                elif np.isin(get_wn_ss(number_of_images[number][1]), get_wn_ss(number_of_images[number][0]).hyponyms()):
                    to_delete.append(number_of_images[number][0])
            if len(number_of_images[number]) > 2:
                f.write(number_of_images[number], number)
                f.write(str(img_ids_to_text(number_of_images[number])) ,number_of_images[number])
    return to_delete


def interest_synsets(image_synset_file_path, min_synset_freq, max_synset_freq):
    """
    This function calculates how many synsets have the appropriate number of images and writes them in a file.
    :return:
    """
    counter = 0
    ss = np.array([])
    try:
        all_hypers = np.load('../data/all_hypers.npz')['ss']
    except:
        all_hypers = generate_all_hypers(image_synset_file_path)
    number_of_images = {}
    for h in all_hypers:
        sume = index_and_hyponims_from_label(h, image_synset_file_path).sum()
        if sume >= min_synset_freq and sume <= max_synset_freq:
            counter += 1
            ss = np.append(ss, h)
            try:
                number_of_images[sume].append(h)
            except:
                number_of_images[sume] = [h]
    to_delete = delete_repeated_ss(number_of_images)
    print('The following synsets will not be used due to redunancy:', to_delete)
    np.savez('../data/to_delete_ss.npz', ss=to_delete)
    np.savetxt('../data/to_delete_ss.txt', to_delete, fmt="%s")
    np.savez('../data/interest_ss.npz', ss=ss)
    print(counter)

    return ss


def extract_interest_synsets(image_synset_file_path, min_synset_freq, max_synset_freq):
    """
    This function iterates over all the synsets that have the appropriate number of images on the dataset (counting hyponyms)
    and generates their partitions.
    :return:
    """
    try:
        all_ss = np.load('../data/all_ss.npz')['ss']
    except:
        all_ss = generate_all_ss(image_synset_file_path)
    try:
        interest = np.load('../data/interest_ss.npz')['ss']
        print('Synset of interest have been loaded')
    except:
        print('Calculating interest synsets')
        interest = interest_synsets(image_synset_file_path, min_synset_freq, max_synset_freq)
        print('Synsets of interest calculated. There are', len(interest), ' synsets')

    counter = 0
    for h in interest:
        counter += generate_partition(h, all_ss, image_synset_file_path, min_synset_freq, max_synset_freq)
    print('We have ', counter, 'partitions of synsets')


def main(image_synset_file_path='../data/imagenet2012_val_synset_codes.txt', min_synset_freq=500,
         max_synset_freq=40000):
    extract_interest_synsets(image_synset_file_path, min_synset_freq, max_synset_freq)


if __name__ == '__main__':
    init = time.time()
    main()
    print('time:', timedelta(seconds=time.time() - init))
