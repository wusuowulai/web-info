import tensorflow as tf
import os
import codecs
import random
import math
import numpy as np
import copy
import time
import re

entityid = {}
headid = {}
relationid = {}

def data_loader(file):
	file1 = file + "train.txt"
	file2 = file + "entity_with_text.txt"
	file3 = file + "relation_with_text.txt"

	with open(file2, 'r') as f1, open(file3, 'r') as f2:
		lines1 = f1.readlines()
		lines2 = f2.readlines()
		for line in lines1:
			line = line.strip().split('\t')
			if len(line) != 2:
				continue
			entityid[line[0]] = line[1]

		for line in lines2:
			line = line.strip().split('\t')
			if len(line) !=2:
				continue
			relationid[line[0]] = line[1]

	entity_set = set()
	relation_set = set()
	triple_list = []

	with codecs.open(file1, 'r') as f:
		content = f.readlines()
		for line in content:
			triple = line.strip().split('\t')
			if len(triple) !=3:
				continue
			triple_list.append(triple)
			entity_set.add(triple[0])
			entity_set.add(triple[2])
			relation_set.add(triple[1])

	return entity_set, relation_set, triple_list

def distanceL2(h,r,t):
	return np.sum(np.square(h + r - t))

def distanceL1(h,r,t):
	return np.sum(np.fabs(h+r-t))






if __name__ == '__main__':
	dir = os.path.dirname(os.path.dirname(__file__))
	dir = dir + '\dataset\lab2_dataset\\'
	entity_set, relation_set, triple_list = data_loader(dir)
	print("load file complete")
	print("entity: %d, relation : %d, triple : %d " % (len(entity_set), len(relation_set), len(triple_list)))

