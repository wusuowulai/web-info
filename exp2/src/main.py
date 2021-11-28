# author:yyk
# version:3.7.2
import os
import pandas as pd
import codecs
import random
import math
import numpy as np
import copy
import time

entity2id = {}
relation2id = {}


def data_loader(dir):
	entity_set = set()
	relation_set = set()
	triple_list= []
	training_file = 'train.txt'
	with open(dir + training_file) as f:
		content = f.readlines()
		for line in content:
			triple = line.strip().split('\t')
			triple_list.append([int(triple[0]), int(triple[2]), int(triple[1])])
	n = 0
	for triple in triple_list:
		if n < triple[0]:
			n = triple[0]
		if n < triple[1]:
			n = triple[1]
	for i in range(int(n) + 1):
		entity_set.add(i)
		entity2id[i] = i
	relation_file = 'relation_with_text.txt'
	relation_df = pd.read_table(os.path.join(dir, relation_file), header=None)
	relation2id = dict(zip(relation_df[0], relation_df[0]))
	for i in range(len(relation2id)):
		relation_set.add(i)
	return entity_set, relation_set, triple_list

def distanceL2(h, r, t):
	# 为方便求梯度，去掉sqrt
	return np.sum(np.square(h + r - t))


def distanceL1(h, r, t):
	return np.sum(np.fabs(h + r - t))


class TransE:
	def __init__(self, entity_set, relation_set, triple_list,
				 embedding_dim=100, learning_rate=0.01, margin=1, L1=True):
		self.embedding_dim = embedding_dim
		self.learning_rate = learning_rate
		self.margin = margin
		self.entity = entity_set
		self.relation = relation_set
		self.triple_list = triple_list
		self.L1 = L1

		self.loss = 0

	def emb_initialize(self):
		relation_dict = {}
		entity_dict = {}

		for relation in self.relation:
			r_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
										   6 / math.sqrt(self.embedding_dim),
										   self.embedding_dim)
			relation_dict[relation] = r_emb_temp / np.linalg.norm(r_emb_temp, ord=2)

		for entity in self.entity:
			e_emb_temp = np.random.uniform(-6 / math.sqrt(self.embedding_dim),
										   6 / math.sqrt(self.embedding_dim),
										   self.embedding_dim)
			entity_dict[entity] = e_emb_temp / np.linalg.norm(e_emb_temp, ord=2)

		self.relation = relation_dict
		self.entity = entity_dict

	def train(self, epochs):
		nbatches = 400
		batch_size = len(self.triple_list) // nbatches
		print("batch size: ", batch_size)
		for epoch in range(epochs):
			start = time.time()
			self.loss = 0

			for k in range(nbatches):
				# Sbatch:list
				# start1 = time.time()
				Sbatch = random.sample(self.triple_list, batch_size)
				Tbatch = []
				for triple in Sbatch:
					corrupted_triple = self.Corrupt(triple)
					Tbatch.append((triple, corrupted_triple))

				self.update_embeddings(Tbatch)
				# end1 = time.time()
				# print('time of one batch: %s'%(round((end1 - start1),3)))
				# return

			end = time.time()
			print("epoch: ", epoch, "cost time: %s" % (round((end - start), 3)))
			print("loss: ", self.loss)

			# 保存临时结果
			if epoch % 10 == 0:
				with codecs.open("..\\res\\entity_temp", "w") as f_e:
					for e in self.entity.keys():
						f_e.write(str(e) + "\t")
						f_e.write(str(list(self.entity[e])))
						f_e.write("\n")
				with codecs.open("..\\res\\relation_temp", "w") as f_r:
					for r in self.relation.keys():
						f_r.write(str(r) + "\t")
						f_r.write(str(list(self.relation[r])))
						f_r.write("\n")
				with codecs.open("..\\res\\result_temp", "a") as f_s:
					f_s.write("epoch: %d\tloss: %s\n" % (epoch, self.loss))

		print("写入文件...")
		with codecs.open("..\\res\\entity_50dim_batch400", "w") as f1:
			for e in self.entity.keys():
				f1.write(str(e) + "\t")
				f1.write(str(list(self.entity[e])))
				f1.write("\n")

		with codecs.open("..\\res\\relation_50dim_batch400", "w") as f2:
			for r in self.relation.keys():
				f2.write(str(r) + "\t")
				f2.write(str(list(self.relation[r])))
				f2.write("\n")
		print("写入完成")

	def Corrupt(self, triple):
		corrupted_triple = copy.deepcopy(triple)
		seed = random.random()
		if seed > 0.5:
			# 替换head
			head = triple[0]
			rand_head = head
			while (rand_head == head):
				rand_head = random.randint(0, len(self.entity) - 1)
			corrupted_triple[0] = rand_head

		else:
			# 替换tail
			tail = triple[0]
			rand_tail = tail
			while (rand_tail == tail):
				rand_tail = random.randint(0, len(self.entity) - 1)
			corrupted_triple[1] = rand_tail
		return corrupted_triple

	def update_embeddings(self, Tbatch):
		# 不要每次都深拷贝整个字典，只拷贝当前Tbatch中出现的三元组对应的向量
		entity_updated = {}
		relation_updated = {}
		for triple, corrupted_triple in Tbatch:
			# 取原始的vector计算梯度
			h_correct = self.entity[triple[0]]
			t_correct = self.entity[triple[1]]
			relation = self.relation[triple[2]]

			h_corrupt = self.entity[corrupted_triple[0]]
			t_corrupt = self.entity[corrupted_triple[1]]
			if triple[0] in entity_updated.keys():
				pass
			else:
				entity_updated[triple[0]] = copy.copy(self.entity[triple[0]])
			if triple[1] in entity_updated.keys():
				pass
			else:
				entity_updated[triple[1]] = copy.copy(self.entity[triple[1]])
			if triple[2] in relation_updated.keys():
				pass
			else:
				relation_updated[triple[2]] = copy.copy(self.relation[triple[2]])
			if corrupted_triple[0] in entity_updated.keys():
				pass
			else:
				entity_updated[corrupted_triple[0]] = copy.copy(self.entity[corrupted_triple[0]])
			if corrupted_triple[1] in entity_updated.keys():
				pass
			else:
				entity_updated[corrupted_triple[1]] = copy.copy(self.entity[corrupted_triple[1]])

			if self.L1:
				dist_correct = distanceL1(h_correct, relation, t_correct)
				dist_corrupt = distanceL1(h_corrupt, relation, t_corrupt)
			else:
				dist_correct = distanceL2(h_correct, relation, t_correct)
				dist_corrupt = distanceL2(h_corrupt, relation, t_corrupt)

			err = self.hinge_loss(dist_correct, dist_corrupt)

			if err > 0:
				self.loss += err
				grad_pos = 2 * (h_correct + relation - t_correct)
				grad_neg = 2 * (h_corrupt + relation - t_corrupt)
				if self.L1:
					for i in range(len(grad_pos)):
						if (grad_pos[i] > 0):
							grad_pos[i] = 1
						else:
							grad_pos[i] = -1

					for i in range(len(grad_neg)):
						if (grad_neg[i] > 0):
							grad_neg[i] = 1
						else:
							grad_neg[i] = -1

				# 梯度求导参考 https://blog.csdn.net/weixin_42348333/article/details/89598144
				entity_updated[triple[0]] -= self.learning_rate * grad_pos
				entity_updated[triple[1]] -= (-1) * self.learning_rate * grad_pos

				entity_updated[corrupted_triple[0]] -= (-1) * self.learning_rate * grad_neg
				entity_updated[corrupted_triple[1]] -= self.learning_rate * grad_neg

				relation_updated[triple[2]] -= self.learning_rate * grad_pos
				relation_updated[triple[2]] -= (-1) * self.learning_rate * grad_neg

		# batch norm
		for i in entity_updated.keys():
			entity_updated[i] /= np.linalg.norm(entity_updated[i])
			self.entity[i] = entity_updated[i]
		for i in relation_updated.keys():
			relation_updated[i] /= np.linalg.norm(relation_updated[i])
			self.relation[i] = relation_updated[i]
		return

	def hinge_loss(self, dist_correct, dist_corrupt):
		return max(0, dist_correct - dist_corrupt + self.margin)


if __name__ == '__main__':
	dir = os.path.dirname(os.path.dirname(__file__))
	dir = dir + '/dataset/lab2_dataset/'
	entity_set, relation_set, triple_list = data_loader(dir)
	print("Complete load. entity : %d , relation : %d , triple : %d" % (len(entity_set), len(relation_set), len(triple_list)))
	transE = TransE(entity_set, relation_set, triple_list, embedding_dim=50, learning_rate=0.01, margin=1, L1=True)
	transE.emb_initialize()
	transE.train(epochs=1001)