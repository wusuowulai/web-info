# author:yyk
# version:3.8.8
import json
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gc

gc.enable()
path0 = format(os.path.abspath(os.path.join(os.getcwd(), "..")))
path = path0 + '\dataset'
path1 = path0 + '\output\\tfidf_matrix.txt'
path2 = path0 + '\output\\name.txt'

FileList = []
data = list()

gc.enable()
# 以下时文件读取部分
for home, dirs, files in os.walk(path):
	for filename in files:
		FileList.append(os.path.join(home, filename))
for file in FileList:
	with open(file, 'r', encoding='utf-8') as j:
		info = json.load(j)
		word2 = info['text']
		word2.join(info['title'])
		# 添加到列表中
		data.append(word2)
		# 存储对应的文件路径以及文件名(后续所有文件通过数字代替进行处理)
		del word2

# 以下时文件处理部分
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
# 生成词频矩阵
count = vectorizer.fit_transform(data)
feature_name = vectorizer.get_feature_names_out()
print(count.shape)
transformer = TfidfTransformer()
# 计算tf-idf
tfidf = transformer.fit_transform(count)

# 存储tf-idf和文件列表
with open(path1, 'wb') as txt:
	pickle.dump(feature_name, txt)
	pickle.dump(tfidf, txt)
	txt.close()
with open(path2, 'wb') as txt:
	pickle.dump(FileList, txt)
	txt.close()


print("请输入查询的词集合")
# 例如输入 “word company”
query = list()
query.append(input())
vectorizer = CountVectorizer(vocabulary=feature_name)
query_list = vectorizer.fit_transform(query)
transformer = TfidfTransformer()
query_tfidf = transformer.fit_transform(vectorizer.fit_transform(query))
similarity = list()
fileindex = list()
for i in range(10):
	similarity.append(1)
	fileindex.append(0)
for j in range(len(FileList)):
	a = str(cosine_similarity(query_tfidf, tfidf[j]).flatten())
	b = float(a.replace('[', '').replace(']', ''))
	if b > min(similarity):
		fileindex[similarity.index(min(similarity))] = j
		similarity[similarity.index(min(similarity))] = b
z = zip(similarity, fileindex)
z = sorted(z, reverse=True)
similarity, fileindex = zip(*z)
for i in range(10):
	print(FileList[fileindex[i]])
	with open(FileList[fileindex[i]], 'r', encoding='utf-8') as j:
		graph = json.load(j)
		print('Graph: ', graph['thread']['main_image'])