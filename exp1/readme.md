### bool检索

1. 代码运行环境 

   win10 CPU：Intel Core i7-10750H with 16GB

   python 3.8.10

   IDE：Visual Studio Code

   **package：re,os,sys,json,pickle,string,spicy.sparse,numpy , nltk,operator**

2. 编译运行方式

   运行python3 bool_search.py，建立过中间文件及倒排表过后，会出现提示用户输入，将想要查询的目标按照规范布尔检索格式输入即可。对应结果将存放至./output/bool_search_result.txt文件，命令行会返回排序后用户设定ser\arch_num个数的原文件路径。


### <center>TF-IDF</center>

1. 代码运行环境
   
   win11 CPU: Intel Core i7-9750H with 16GB

   python 3.8.8 (anaconda3 64bit)

   IDE: pycharm

   **package: json, pickle, os, numpy, skicit-learn**

2. 编译运行方式
   直接运行[semantic_search.py](./src/semantic_search.py)即可，首先进行数据读取，然后按照提示输入要查询的词的集合

3. 运行所需空间
   助教所给测试集文件读入五个文件夹，读取数据并生成矩阵时占用3GB内存，生成的数据文件大小为600MB， 最后查找相似文件时占用空间2.3GB

   
