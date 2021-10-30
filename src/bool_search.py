
import re
import os
import sys
import json
import pickle
import string
import operator

from scipy.sparse import coo_matrix, hstack
import numpy as np

import nltk
import nltk.stem
from nltk import *
from nltk.corpus import stopwords
from nltk.tokenize import WhitespaceTokenizer

search_num = 10
# bool检索命令行返回相关度前10的文档


path1 = './dataset/US_Financial_News_Articles/'
#os.makedirs("./WordSegmentation_US_Financial_News_Articles/")
#print(ifmkdir)
#path2 = './WordSegmentation_US_Financial_News_Articles/'
path3 = './'

stem_snowball = nltk.stem.SnowballStemmer('english')


# 返回path包含文件或文件夹名字的列表
file_list1 = os.listdir(path1)
stopwords = stopwords.words('english')

table = {}
    
for file_name1 in file_list1:

    file_list1_2 = os.listdir(path1 + file_name1)

    for file_name2 in file_list1_2:

        with open(path1 + file_name1 + '/' + file_name2, 'r', encoding='utf-8') as j:
            info = json.load(j)
            #print(file_name2)
            word2 = info['text']
            # file存储当前读取的文件名称

            # 改进缩写
            word1 = info['text'].replace('won\'t', 'will not').replace('I\'m', 'I am').replace('\'s', 
                ' is').replace('\'d','would').replace('can\'t', 'cannot')
            word1 = word1.replace('\'t', ' not').replace('\'ve', ' have').replace('\'re', ' are').replace('-', '').replace(
                'is\'t', 'is not').replace('\'ll', ' will')
            text = str(word1)
            text1 = text.lower()
            # print(text1)
            word1 = WhitespaceTokenizer().tokenize(text1)
            # nltk进行分词
            words = [word for word in word1 if word not in stopwords]
            # 去除停止词
            # print(word1)
            regex_punctuation = re.compile('[%s]' % re.escape(string.punctuation))
            new_words = filter(lambda word: word != '', [regex_punctuation.sub('', word) for word in words])
            # 去除标点符号

            new_words_list = list(new_words)

            for i,i_words in enumerate(new_words_list):
                stem_snowball = nltk.stem.SnowballStemmer('english')
                i_words = stem_snowball.stem(i_words)
                new_words_list[i] = i_words
                #print(i_words)

            # 建立倒排索引表

            # 把经过wordSegmentation的文件整理起来
            # 读这些文件，构建表

            for words in new_words_list:
                
                # 构建index table
                # table中没有words，返回默认值None
                if table.get(words) == None:
                    table[words] = {file_name1+'.'+file_name2:1}
                    print(file_name1+'.'+file_name2)
                # table中有words
                else:
                    # 未出现过该文档ID
                    if table[words].get(file_name1+'.'+file_name2) == None:
                        table[words][file_name1+'.'+file_name2] = 1
                    else:
                        #print(table[words][file_list2[i]])
                        table[words][file_name1+'.'+file_name2] = table[words][file_name1+'.'+file_name2] + 1




print("creating an inverted Index table")    
                
# 输出结果
output = open(path3 + 'output/IndexTable.txt','w')
output.write(json.dumps(table))
output.close()

#############################################################


file = open(path3 + 'output/IndexTable.txt','r')
IndexTable = file.read()
IndexTable = json.loads(IndexTable)

#print('?')

print("Congratulations！ Index Table 生成成功")


import ply.lex as lex

tokens = (
    'NAME',
    'AND',
    'OR',
    'NOT',
    'LPAREN',
    'RPAREN',
)

t_AND = r'AND'
t_OR = r'OR'
t_NOT = r'NOT'
t_LPAREN = r'\('
t_RPAREN = r'\)'

def t_NAME(t):
    r'[a-z0-9]+'
    #print(t.value)
    #print(t.type)
    return t

t_ignore = '[\t ]'

def t_error(t):
    print("输入格式错误哦~")
    t.lexer.skip(1)

lexer = lex.lex()

'''
data = 'apple AND banana'
lexer.input(str.lower(data).replace(' and ',' AND ').replace(' or ',' OR ').replace(' not ',' NOT '))

while True:
    tok = lexer.token()
    if not tok: break
    print(tok)
'''

import ply.yacc as yacc
  
def p_result(p):
    '''
    result : LISTOR
    '''
    global result
    p[0] = p[1]
    result = p[0]

def p_list_or(p):
    '''
    LISTOR : LISTOR OR LISTAND
        | LISTAND
    '''
    if(p.slice[1].type=="LISTOR"):
        p[0] = {}
        for key in p[1]:
            if p[3].get(key) != None:
                p[0][key]=p[1][key]+p[3][key]
            else:
                p[0][key]=p[1][key]+0
        for key in p[3]:
            if p[0].get(key) == None:
                p[0][key]=p[3][key]
    else:
        p[0] = p[1]
    

def p_list_add(p):
    '''
    LISTAND : LISTAND AND LIST
        | LIST
    '''
    #print("???")
    if(p.slice[1].type=="LISTAND"):
        p[0] = {}
        for key in p[1]:
            #print(key,'!!!')
            if p[3].get(key) != None:
                p[0][key]=p[1][key]+p[3][key]
    else:
        p[0] = p[1]
    
    
def p_list(p):
    '''
    LIST : LPAREN LISTOR RPAREN
        | NAME
        | NOT NAME
    '''
    #print('3',p.slice[1].type)
    if(p.slice[1].type == 'NAME'):
        keyword = p.slice[1].value

        #stemming,获取倒排表中相应内容
        keyword = stem_snowball.stem(keyword)     
        keyword_value = IndexTable.get(keyword)

        if keyword_value == None:
            print("Please don't type the stop word",keyword,"，I'm just going to ignore it！！")
            p[0] = {}
        else:
            p[0] = keyword_value

    elif(p.slice[1].type == 'NOT'):
        keyword = p.slice[1].value

        p[0]={}

        #stemming,获取倒排表中相应内容
        keyword = stem_snowball.stem(keyword)
        keyword_dict = IndexTable.get(keyword)

        for file_name1 in file_list1:

            file_list1_2 = os.listdir(path1 + file_name1)
            for file_name2 in file_list1_2:

                if keyword_dict.get(file_name1+'.'+file_name2) == None:
                    p[0][file_name1+'.'+file_name2] = 1
                    #print(file_name1+'.'+file_name2)
    else:
        p[0] = p[2]
    #print('33',p.slice[0].type)


def p_error(p):
    print("error")

parser = yacc.yacc()


print('What do you wanna search for?')
input = input('Please enter your imformation in bool mode:')
parser.parse(input.replace(' and ',' AND ').replace(' or ',' OR ').replace(' not ',' NOT '))

result_key=[ key for key,value in result.items()] 

for i,i_paths in enumerate(result_key):

    i_paths = path1 + i_paths.replace('.','/',1)
    result_key[i] = i_paths
    #print(result_key[i])

str = '\n'
search_result_file = open(path3+"output/bool_search_result.txt",'w')
search_result_file.write(str.join(result_key))
search_result_file.close()

# 按照键值排序
result = sorted(result.items(), key=lambda d:d[1], reverse = True)
result = dict(result)

result = [ key for key,value in result.items()] 
num = 0
for i_paths in result:
    if num == search_num:
        break
    num = num + 1
    print(i_paths.replace('.','/',1))
