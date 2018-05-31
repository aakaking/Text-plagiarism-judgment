
# coding: utf-8

# # 数据基本信息

# In[1]:


import pandas as pd


# In[2]:


content = pd.read_csv('D:/Code/Pycharm/Data_source/sqlResult_1558435.csv',encoding='gb18030')


# In[3]:


len(content)


# In[4]:


content.columns


# In[5]:


content.head()


# In[6]:


xinhua_data = []
for i in range(len(content)):
    xinhua_data.append(1 if '新华' in str(content.iloc[i].source) else 0)
    xinhua_data.count(1)


# # 文本自动聚类（新华社文章）

# ### 停用词

# In[7]:


def get_stopwords(filename = "D:/Code/Pycharm/Data_source/chinese_stopwords.txt"):
    stopwords_dic = open(filename, encoding= 'utf-8')
    stopwords = stopwords_dic.readlines()
    stopwords = [w.strip() for w in stopwords]
    stopwords_dic.close()
    print(stopwords)
    stopwords.append('\r\n')
    return stopwords


# In[8]:


stopwords = get_stopwords()


# ### 准备数据

# In[9]:


import jieba


# In[10]:


def cut(string): return list(jieba.cut(string))


# In[11]:


def clean(words):
    clean_words = []
    for word in words:
        if not word.isdigit() and word not in stopwords and 1<len(word)<5:
            clean_words.append(word)
    return clean_words


# In[12]:


train_data_list = []
data_id = []
for i in range(len(content)):
    if '新华' in str(content.iloc[i].source):
        train_data_list.append(" ".join(clean(cut(str(content.iloc[i].content)))))
        data_id.append(i)


# ### 建立TFIDF向量

# In[13]:


from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer 


# In[14]:


count_v1 = CountVectorizer(max_df=0.4, max_features=3000) # 考虑到内存只选用了3000个特征
counts_train = count_v1.fit_transform(train_data_list)
print("the shape of train is " + repr(counts_train.shape))
tfidftransformer = TfidfTransformer()
tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
tfidf_ndarray = tfidf_train.toarray() 


# ### 用K-Means聚类

# In[15]:


from sklearn.cluster import KMeans


# In[16]:


kmeans = KMeans(n_clusters=10, random_state=0).fit(tfidf_ndarray)
label = kmeans.labels_


# ### 建立 id==>kinds 和 kind==>ids 映射

# In[17]:


id_kinds = {text_id: kind_id for text_id, kind_id in zip(data_id,kmeans.labels_)}


# In[18]:


id_kinds[3]


# In[19]:


from collections import defaultdict
kind_ids = {}
kind_ids = defaultdict(lambda: set())
for text_id, kind_id in id_kinds.items():kind_ids[kind_id].add(text_id)


# In[20]:


kind_ids[1]


# In[21]:


for i in range(10):
    print(len(kind_ids[i]))


<<<<<<< HEAD
# ### 词云展示聚类结果
=======
# ### 词云展示分类结果
>>>>>>> 74c59a1d2a37a18bf2b3b6f2eaf0ee4aa2be16ed

# In[55]:


select_kind = kind_ids[2]

from collections import Counter
kind_words_counter = [Counter(clean(cut(str(content.iloc[num].content)))) for num in list(select_kind)]
kind_all_word = kind_words_counter[0] 
for i in range(1,len(select_kind)):kind_all_word += kind_words_counter[i] 
cloud_data = ''
for num in list(select_kind):
    cloud_data = cloud_data + ' ' + (" ".join(clean(cut(str(content.iloc[num].content)))))
    
import wordcloud 
wc = wordcloud.WordCloud(font_path=r'C:\Users\Anan\Downloads/simkai.ttf',
                         stopwords='新华社', width=800, height=500, scale=2, 
                         max_words=200, background_color='black')
#word_cloud = wc.generate_from_text(cloud_data)
word_cloud = wc.generate_from_frequencies(kind_all_word)

import matplotlib.pyplot as plt
plt.imshow(word_cloud)


# # 文本分类（判断是否为新华社所发）

# ### 数据打标签

# In[23]:


data_list = []
class_list = []
for i in range(len(content)):
    data_list.append(" ".join(clean(cut(str(content.iloc[i].content)))))
    class_list.append(1 if '新华' in str(content.iloc[i].source) else 0)


# ### 随机选取训练数据和测试数据

# In[24]:


import random


# In[25]:


test_size = 0.2    
data_index = [i for i in range(len(content))]    
data_class_list = list(zip(data_list, class_list, data_index))
random.shuffle(data_class_list)
index = int(len(data_class_list)*test_size) + 1
train_list = data_class_list[index:]
test_list = data_class_list[:index]
train_data_list, train_class_list, train_index_list = zip(*train_list)
test_data_list, test_class_list, test_index_list = zip(*test_list)


# ### 建立TFIDF向量

# In[26]:


count_clf = CountVectorizer(max_df=0.5)
counts_train = count_clf.fit_transform(train_data_list)

count_v2 = CountVectorizer(vocabulary=count_clf.vocabulary_)
counts_test = count_v2.fit_transform(test_data_list)

tfidftransformer = TfidfTransformer()

tfidf_train = tfidftransformer.fit(counts_train).transform(counts_train)
tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)
print("the shape of tfidf_train is " + repr(tfidf_train.shape))
print("the shape of tfidf_test test is " + repr(tfidf_test.shape))


# ### 使用朴素贝叶斯分类器

# In[27]:


import pickle


# In[28]:


def NBClassifier(train_feature, train_label, test_feature):
    from sklearn.naive_bayes import MultinomialNB
    nbclf = MultinomialNB().fit(train_feature, train_label)
    filename = 'finalized_nbclf.sav'
    pickle.dump(nbclf, open(filename, 'wb')) 
    pred = nbclf.predict(test_feature)
    return pred


# ### 性能评估

# In[31]:


from sklearn import metrics


# In[32]:


def calculate_result(actual,pred):
    m_accuracy = metrics.accuracy_score(actual, pred)
    m_precision = metrics.precision_score(actual,pred)
    m_recall = metrics.recall_score(actual,pred)
    print('predict info:')
    print('accuracy:{0:.3f}'.format(m_accuracy))
    print('precision:{0:.3f}'.format(m_precision))
    print('recall:{0:0.3f}'.format(m_recall))
    print('f1-score:{0:.3f}'.format(metrics.f1_score(actual,pred)))


# ### 结果显示

# In[33]:


pred = NBClassifier(tfidf_train, train_class_list, tfidf_test) 
print('*************************\nNaiveBayes\n*************************')
calculate_result(test_class_list, pred)


# ### 找出可能抄袭的文章

# In[34]:


import numpy as np


# In[35]:


result = []
result = [test_index_list[i] for i in range(len(pred)) if list(pred)[i]==1 and list(test_class_list)[i]!=1]
np.save('result.npy', result)
result


# In[36]:


len(result)


# # 抄袭判定（利用余弦距离计算相似性）

# In[37]:


def CosineDistance(x,y):
    result = np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))
    return result


# ### 准备数据

# In[38]:


test_data_list = []
for i in result:
    test_data_list.append( " ".join(clean(cut(str(content.iloc[i].content)))))
count_v2 = CountVectorizer(vocabulary=count_v1.vocabulary_)
counts_test = count_v2.fit_transform(test_data_list)
tfidf_test = tfidftransformer.fit(counts_test).transform(counts_test)
test_ndarray = tfidf_test.toarray()
print("the shape of tfidf_test is " + repr(tfidf_test.shape))


# ### 计算余弦距离

# In[39]:


Distance = []
for i in range(len(result)):
    distance_for_one = []
    for k in range(len(tfidf_ndarray)):
        distance_for_one.append(CosineDistance(test_ndarray[i][:],tfidf_ndarray[k][:]))
    Distance.append([max(distance_for_one),result[i],data_id[distance_for_one.index(max(distance_for_one))]])


# ### 观察结果

# In[40]:


Distance[1:10]


# ### 确定阈值 选出相似性较高的样本

# In[41]:


[element for element in Distance if element[0] > 0.6]


# ### 利用 editdistance 精确定位

# In[42]:


import editdistance


# In[43]:


import re


# In[44]:


def cut2sentence(string): return re.split(u'，|。|；|、|？|', string)


# In[45]:


def get_edit_distance(str1, str2): return editdistance.eval(cut(str1), cut(str2))


# In[46]:


def get_content(num): return content.iloc[num].content


# In[47]:


string1 = cut2sentence(get_content(7525))
string2 = cut2sentence(get_content(29376)) #xinhua29376


# In[49]:


ed = []
for k in range(len(string1)):
    for i in range(len(string2)):
        if get_edit_distance(string1[k], string2[i]) == 0:
            ed.append([k,i])


# In[50]:


ed


# In[54]:


for element in ed:
    print(string1[element[0]], string2[element[1]])

