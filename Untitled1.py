#!/usr/bin/env python
# coding: utf-8

# In[9]:


#pip install rake-nltk


from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import nltk 
from nltk.tokenize import word_tokenize, sent_tokenize 
import re

stop_words = set(stopwords.words('english')) 


def read_article(file_name):
    file = open(file_name, "r", encoding = 'utf8')
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []

    for sentence in article:
        #print(sentence)
        sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    sentences.pop() 
    
    return sentences


 
def build_similarity_matrix(sentences, stop_words):
    
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
 
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2: 
                continue 
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)

    return similarity_matrix

def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
 
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
 
    all_words = list(set(sent1 + sent2))
 
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
 
    
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
 
   
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
 
    return 1 - cosine_distance(vector1, vector2)

def untokenize(words):
    
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()



def generate_summary(file_name, n):
    stop_words = stopwords.words('english')
    summarize_text = []
   
    sentences =  read_article(file_name)

    
    sentence_similarity_martix = build_similarity_matrix(sentences, stop_words)

  
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_martix)
    scores = nx.pagerank(sentence_similarity_graph)

  
    ranked_sentence = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)    
    print("Indexes of top ranked_sentence order are ", ranked_sentence)    
    
    
    
    for i in range(n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
 
    
    
    return ranked_sentence

summ = generate_summary( "msft.txt", 2)
#textj = " this the is the world we livein the major form"
#wordsjj = word_tokenize(textj)
#print("from  here")
#print(wordsjj)


# In[10]:






question = []



for r in range(len(summ)):
    question.append(" ")
    question[r] = summ[r][1]
        
    
    
    

#print(question)   

for i in range(len(summ)):
        print(question[i])


# In[11]:


from rake_nltk import Rake


# In[ ]:




 


# In[12]:


r = Rake() # Uses stopwords for english from NLTK, and all puntuation characters.
#question[1] 
ranked_q = []
#print(question[1])
for i in range(len(summ)):
    
    r.extract_keywords_from_text(untokenize(question[i]))
    ranked_q.append(r.get_ranked_phrases())
    
    #ranked_q = r.get_ranked_phrases() # To get keyword phrases ranked highest to lowest.
for i in range (len(ranked_q)):
    print(untokenize(question[i]))
    print(ranked_q[i])


# In[13]:


#print(question[1])
selected_k = []
for r in range(len(ranked_q)):
    pos = nltk.pos_tag(ranked_q[r])
    selective_pos = ['NN','VB']
    selective_pos_words = []
    for word,tag in pos:
     if tag in selective_pos:
         selective_pos_words.append((word,tag))
    selected_k.append(selective_pos_words)
#print(selected_k[1][0])


  
   

    
    
    





# In[14]:


from nltk.tokenize import SpaceTokenizer
tm = SpaceTokenizer()
to_rank = []
key_words = []

for i in range (len(ranked_q)) : 
    yn = 0
    
    #ranked_q[i][yn]
    question[i]= untokenize(question[i])
    
    yy = "_____"
    to_rank.append(tm.tokenize(ranked_q[i][0]))
    print("Q:",question[i].replace(to_rank[i][len(to_rank[i])//2] ,yy))
    print('Ans - ',to_rank[i][len(to_rank[i])//2])
    #quita = question[i].index(to_rank[i][len(to_rank[i])//2])
    
    #key_words.append(question[i][quita])
    
#print(to_rank[0][len(to_rank[0])//2])   

#question[0].remove(question[0][quita])

#question[0][quita] = to_rank[0][len(to_rank[0])//2]
#print(question[0][quita])



  
# In[ ]:





# In[ ]:





# In[ ]:




