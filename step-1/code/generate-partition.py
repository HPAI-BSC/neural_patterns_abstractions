"""
In this code we generate the partition of all the synsets with at least 1000 images.

For faster calculations it will generate the next auxiliar files:
	all_ss : The set of synsets that appear in the dataset.
	all_hypers: The set of hypernims of all_ss
	interest_ss: The set of synsets that have more than 1000 images including hyponyms.
	imagenet.....npz: The information of the dataset separated on two numpy array.
"""

from nltk.corpus import wordnet as wn
import numpy as np
from datetime import timedelta
import time
import os
import functools as fu


def ss_to_text(synset):
	""" Returns the name of a given synset"""
	return str(synset)[8:-7]

def get_wn_ss(imagenet_id):
	"""
	Transforms an imagenet id into a wordnet synset
	:param imagenet_id:
	:return:
	"""
	from nltk.corpus import wordnet as wn
	return wn.of2ss(imagenet_id[1:] + '-' + imagenet_id[0])

def img_ids_to_text(samples):
	"""
	Transforms an array of imagenet ids to an array of its names
	:param samples:
	:return:
	"""
	ss = [ss_to_text(get_wn_ss(k)) for k in samples]
	return ss

def get_in_id( wordnet_ss):
	"""
	Transforms a worndet synset into a imagenet id
	Input: Synset
	:param wordnet_ss:
	:return: imagenet id (string)
	"""
	from nltk.corpus import wordnet as wn
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

def generate_all_ss():
	"""
	Generates a file with all the diferent synsets that appear the imagenet2012_val_synset_codes file without repetitions.
	:return:
	"""
	all_ss = np.array([])
	with open('../data/imagenet2012_val_synset_codes.txt', 'r') as f:
		for l in f:
			code = l.strip().split()[1]
			if np.isin(code, all_ss).sum():
				pass
			else:
				all_ss = np.append(all_ss, code)

	np.savez('../data/all_ss.npz', ss=all_ss)
	return  all_ss

def generate_all_hypers():
	"""
	Generates a file with all the hyponyms of the set of synsets that appear in the imagenet2012_val_synset_codes file (without repetitions).
	:return:
	"""
	try:
		all_ss = np.load('../data/all_ss.npz')['ss']
	except:
		all_ss = generate_all_ss()

	all_hypers = np.array([])
	for ss in all_ss:
		all_hypers = np.append(all_hypers,ss)
		hypers = obtain_hypernyms(ss)
		for h in hypers:
			if np.isin(h, all_hypers).sum():
				pass
			else:
				all_hypers = np.append(all_hypers, h)

	print(len(all_hypers),all_hypers)
	np.savez('../data/all_hypers.npz', ss=all_hypers)
	return all_hypers

def generate_all_codes_and_images_as_array():
	"""
	Creates a npz with the information of imagenet2012_val_synset_codes.
	It's divided two arrays:
	- all_imgs: the names of the images of the original file.
	- all_ss: the imagenet id of the class of each image.
	:return:
	"""
	all_ss = np.array([])
	all_imgs = np.array([])
	with open('../data/imagenet2012_val_synset_codes.txt', 'r') as f:
		for l in f:
			code = l.strip().split()[1]
			img = l.strip().split()[0]
			all_ss = np.append(all_ss, code)
			all_imgs = np.append(all_imgs, img)

	np.savez('../data/imagenet2012_val_synset_codes_as_array.npz', ss=all_ss,imgs=all_imgs)
	return  all_ss,all_imgs


def index_and_hyponims_from_label(ss):
	"""
	This function generates an index of the position of the synset or its hyponims in the format:
	index[x] = 1 if the synset ss or one of its hyponyms is in position x.
	index[0] = 0 else.
	:param ss:
	:return:
	"""
	try:
		all_labels = np.load('../data/imagenet2012_val_synset_codes_as_array.npz')['ss']
	except:
		all_labels = generate_all_codes_and_images_as_array()[0]
	try:
		dif_labels = np.load('../data/all_ss.npz')['ss']
	except:
		generate_all_ss()
	hyp = np.array(obtain_hyponims(ss))
	hyp_in_labels = np.isin(hyp, dif_labels)
	hyp = hyp[hyp_in_labels]
	hyp = np.append(ss, hyp)
	indexs = [k == all_labels for k in hyp]
	index = fu.reduce(lambda a, b: a + b, indexs)
	return index


def generate_partition(goal_synset):
	"""
	This function makes the partition of the images with the criteria if is goal_synset (or hyponym) or not.
	And writes two files with this partion in ../data/goal_synset/ if there are more than 1000 images of this synset.
	"""
	ss = np.array([])
	no_ss = np.array([])

	with open('../data/imagenet2012_val_synset_codes.txt', 'r') as f:
		for l in f:
			code = l.strip().split()[1]
			hypernyms = obtain_hypernyms(code)

			if goal_synset in hypernyms:
				ss = np.append(ss, l.strip().split()[0])
			else:
				no_ss = np.append(no_ss, l.strip().split()[0])

	if len(ss) > 1000:
		try:
			os.mkdir('../data/' + goal_synset)
		except:
			pass
		np.savetxt('../data/' + goal_synset +'/' + img_ids_to_text([goal_synset])[0] + '_images.txt' , ss,  fmt="%s")
		np.savetxt('../data/' + goal_synset + '/no_'+ img_ids_to_text([goal_synset])[0] + '_images.txt', no_ss, fmt="%s")
		return 1
	return 0


def interest_synsets():
	"""
	This function calculates how many synsets have more than 1000 images and writes them in a file.
	(54)
	:return:
	"""
	counter = 0
	ss = np.array([])
	try:
		all_hypers  = np.load('../data/all_hypers.npz')['ss']
	except:
		all_hypers = generate_all_hypers()

	for h in all_hypers:
		if index_and_hyponims_from_label(h).sum()>1000:
			counter += 1
			ss = np.append(ss, h)
	np.savez('../data/interest_ss.npz', ss=ss)
	print(counter)

	return ss


def extract_interest_synsets():
	"""
	This function iterates over all the synsets that appear more than 1000 times on the dataset (counting hyponims)
	and generates their partitions.
	:return:
	"""
	counter = 0
	try:
		interest  = np.load('../data/interest_ss.npz')['ss']
	except:
		interest = interest_synsets()
	for h in interest:
		counter += generate_partition(h)
	print(counter)

def main():
	extract_interest_synsets()

if __name__ == '__main__':
	init = time.time()
	main()
	print('time:', timedelta(seconds=time.time() - init))