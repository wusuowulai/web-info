# author:yyk
# version:3.7.2
import numpy as np
import codecs
import operator
import os
import json
from main import data_loader, entity2id, relation2id
import time


def dataloader(entity_file, relation_file, test_file):
	# entity_file: entity \t embedding
	entity_dict = {}
	relation_dict = {}
	test_triple = []

	with codecs.open(entity_file) as e_f:
		lines = e_f.readlines()
		for line in lines:
			entity, embedding = line.strip().split('\t')
			embedding = np.array(json.loads(embedding))
			entity_dict[int(entity)] = embedding

	with codecs.open(relation_file) as r_f:
		lines = r_f.readlines()
		for line in lines:
			relation, embedding = line.strip().split('\t')
			embedding = np.array(json.loads(embedding))
			relation_dict[int(relation)] = embedding

	with codecs.open(test_file) as t_f:
		lines = t_f.readlines()
		for line in lines:
			triple = line.strip().split('\t')
			if len(triple) != 3:
				continue
			h_ = entity2id[int(triple[0])]
			t_ = entity2id[int(triple[2])]
			r_ = relation2id[int(triple[1])]

			test_triple.append(tuple((h_, t_, r_)))

	return entity_dict, relation_dict, test_triple


def distance(h, r, t):
	return np.linalg.norm(h + r - t)


class Test:
	def __init__(self, entity_dict, relation_dict, test_triple, train_triple, isFit=True):
		self.entity_dict = entity_dict
		self.relation_dict = relation_dict
		self.test_triple = test_triple
		self.train_triple = train_triple
		print(len(self.entity_dict), len(self.relation_dict), len(self.test_triple), len(self.train_triple))
		self.isFit = isFit

		self.hits10 = 0
		self.mean_rank = 0

		self.relation_hits10 = 0
		self.relation_mean_rank = 0

	def rank(self):
		hits = 0
		hit1 = 0
		rank_sum = 0
		rank = 0
		step = 1
		start = time.time()
		for triple in self.test_triple:
			rank_head_dict = {}
			rank_tail_dict = {}

			for entity in self.entity_dict.keys():

				if self.isFit:
					if [triple[0], entity, triple[2]] not in self.train_triple:
						h_emb = self.entity_dict[triple[0]]
						r_emb = self.relation_dict[triple[2]]
						t_emb = self.entity_dict[entity]
						rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)
				else:
					h_emb = self.entity_dict[triple[0]]
					r_emb = self.relation_dict[triple[2]]
					t_emb = self.entity_dict[entity]
					rank_tail_dict[entity] = distance(h_emb, r_emb, t_emb)

			rank_tail_sorted = sorted(rank_tail_dict.items(), key=operator.itemgetter(1))

			for i in range(len(rank_tail_sorted)):
				if triple[1] == rank_tail_sorted[i][0]:
					if i < 5:
						hits += 1
					if i < 1:
						hit1 += 1
					rank_sum = rank_sum + i + 1
					rank = rank + i + 1
					break

			step += 1
			if step % 200 == 0:
				end = time.time()
				print("step: ", step, " ,hit_top5_rate: ", hits / step, " ,mean_rank ", rank_sum / step,
					  'time of testing one triple: %s' % (round((end - start), 3)))
				print("step: ", step, ", hit_1_rate: ", hit1 / step, " mean_rank ", rank / step)
				start = end
		self.hits10 = hits / (len(self.test_triple))
		self.mean_rank = rank_sum / (len(self.test_triple))

	def relation_rank(self):
		hits = 0
		rank_sum = 0
		step = 1

		start = time.time()
		for triple in self.test_triple:
			rank_dict = {}
			for r in self.relation_dict.keys():
				if self.isFit and (triple[0], triple[1], r) in self.train_triple:
					continue
				h_emb = self.entity_dict[int(triple[0])]
				r_emb = self.relation_dict[int(r)]
				t_emb = self.entity_dict[int(triple[1])]
				rank_dict[r] = distance(h_emb, r_emb, t_emb)

			rank_sorted = sorted(rank_dict.items(), key=operator.itemgetter(1))

			rank = 1
			for i in rank_sorted:
				if triple[2] == i[0]:
					break
				rank += 1
			if rank < 10:
				hits += 1
			rank_sum = rank_sum + rank + 1

			step += 1
			if step % 200 == 0:
				end = time.time()
				print("step: ", step, " ,hit_top10_rate: ", hits / step, " ,mean_rank ", rank_sum / step,
					  'used time: %s' % (round((end - start), 3)))
				start = end

		self.relation_hits10 = hits / len(self.test_triple)
		self.relation_mean_rank = rank_sum / len(self.test_triple)


if __name__ == '__main__':
	dir = os.path.dirname(os.path.dirname(__file__))
	dir1 = dir + '/dataset/lab2_dataset/'
	_, _, train_triple = data_loader(dir1)

	entity_dict, relation_dict, test_triple = \
		dataloader(dir + "\\res\\entity_50dim_batch400", dir + "\\res\\relation_50dim_batch400",
				   dir + "\\dataset\\lab2_dataset\\test.txt")

	test = Test(entity_dict, relation_dict, test_triple, train_triple, isFit=False)

	test.rank()
	print("entity hits@10: ", test.hits10)
	print("entity meanrank: ", test.mean_rank)

