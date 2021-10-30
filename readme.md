### bool检索

1. 代码运行环境 

   win10 CPU：Intel Core i7-10750H with 16GB

   python 3.8.10

   IDE：Visual Studio Code

   **package：re,os,sys,json,pickle,string,spicy.sparse,numpy,nltk,operator**

2. 编译运行方式

   运行python3 bool_search.py，建立过中间文件及倒排表过后，会出现提示用户输入，将想要查询的目标按照规范布尔检索格式输入即可。对应结果将存放至./search_result.txt文件，命令行会返回排序后用户设定num个数的原文件路径。

3. 运行所需空间

   



### <center>TF-IDF</center>

1. 代码运行环境
   win11 CPU: Intel Core i7-9750H with 16GB
   python 3.8.8 (anaconda3 64bit)
   IDE: pycharm
   **package: pickle, os, numpy, sklearn.feature_extraction.text, sklearn.metrics.pairwise**

2. 编译运行方式
   首先运行[build_matrix.py](./src/semantic_search/build_matrix.py)用于读取数据和生成tf-idf矩阵，然后后运行[search.py](./src/semantic_search/search.py),需要输入所需的测试集合，输入格式为`word1 word2 word3 ···`然后会输出结果

3. 运行所需空间
   助教所给测试集文件读入五个文件夹，读取数据并生成矩阵时占用3GB内存，生成的数据文件大小为600MB， 最后查找相似文件时占用空间2.3GB

   
