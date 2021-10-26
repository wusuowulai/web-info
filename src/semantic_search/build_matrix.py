# author:yyk
# version:3.8.8
import json
import os
import gc
import pickle
import datetime
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

start = datetime.datetime.now()

gc.enable()
path0 = format(os.path.abspath(os.path.join(os.getcwd(), "../..")))
path = path0 + '\dataset'
path1 = path0 + '\output\\tfidf_matrix.txt'
path2 = path0 + '\output\\name.txt'

FileList = []
data = list()

# 以下时文件读取部分
for home, dirs, files in os.walk(path):
	for filename in files:
		FileList.append(os.path.join(home, filename))
for file in FileList:
	with open(file, 'r', encoding='utf-8') as j:
		info = json.load(j)
		word2 = info['text']
		# 添加到列表中
		data.append(word2)
		# 存储对应的文件路径以及文件名(后续所有文件通过数字代替进行处理)
		del word2

# 以下时文件处理部分
vectorizer = CountVectorizer(stop_words='english', lowercase=True)
count = vectorizer.fit_transform(data)
feature_name = vectorizer.get_feature_names_out()
print(count.shape)
transformer = TfidfTransformer()
tf_idf = transformer.fit_transform(vectorizer.fit_transform(data))

with open(path1, 'wb') as txt:
	pickle.dump(feature_name, txt)
	pickle.dump(tf_idf, txt)
	txt.close()
with open(path2, 'wb') as txt:
	pickle.dump(FileList, txt)
	txt.close()

end = datetime.datetime.now()
print(end-start)