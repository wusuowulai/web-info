# author:yyk
# version:3.7.2
import json
import pickle
import os
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import gc

path0 = format(os.path.abspath(os.path.join(os.getcwd(), "../..")))
path1 = path0 + '\output\\tfidf_matrix.txt'
path2 = path0 + '\output\\name.txt'
start = datetime.datetime.now()
gc.enable()
with open(path1, 'rb') as txt:
	feature_name = pickle.load(txt)
	tfidf = pickle.load(txt)
	txt.close()
with open(path2, 'rb') as txt:
	filename = pickle.load(txt)
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
	similarity.append(0)
	fileindex.append(0)
for j in range(len(filename)):
	a = str(cosine_similarity(query_tfidf, tfidf[j]).flatten())
	b = float(a.replace('[', '').replace(']', ''))
	if b > min(similarity):
		fileindex[similarity.index(min(similarity))] = j
		similarity[similarity.index(min(similarity))] = b
z = zip(similarity, fileindex)
z = sorted(z, reverse=True)
similarity, fileindex = zip(*z)
for i in range(10):
	print(filename[fileindex[i]], ': ', similarity[i])
	with open(filename[fileindex[i]], 'r', encoding='utf-8') as j:
		graph = json.load(j)
		print('Graph: ', graph['thread']['main_image'])

end = datetime.datetime.now()
print(end -start)
