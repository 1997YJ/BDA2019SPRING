# =============================================================================
# <                   BDA Homework1  - Find Keywords                         >
#  FileName     [ find_keywords.py ]
#  Authors      [ Yen-Jung Hsu ]
#  Date         [ 2019/3/11 created ]
# =============================================================================
import csv
import codecs
import numpy as np
import pandas as pd
from collections import Counter
# =============================================================================
# Read Stopwords Documents
# =============================================================================
stopwords = []
with open('./ref/stopwords.txt', 'r', encoding='UTF-8') as file:
    for each in file.readlines():
        stopwords.append(each.strip())  
# =============================================================================
# Remove puntuations, numbers and alphbets
# =============================================================================
def remove_puntuations(content_strings,user_pun = False):
    if(user_pun):
        punctuation = user_pun
    else:
        punctuation = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890ㄅㄆㄇㄈㄉㄊㄋㄌㄍㄎㄏㄐㄑㄒㄓㄔㄕㄖㄗㄘㄙㄧㄨㄩㄚㄛㄜㄝㄞㄟㄠㄡㄢㄣㄤㄥㄦ! ^_`{|}~\ ©™…—–‑    ©®™§¶…‘’“”—ˊˋˇΣαβγπ–×––〝·°—“” ‘’㏄· ※℃。）〞〞∕〈〉〈〉（╱ √≠╳█■□▽○％◎●★—‧─ ─．˙…！／【】『』〖〗〔〕:;!@#$%^&*()?_+=-[]`~'\"|/\\ˊˇˋ˙,.{}ⅠⅡⅢⅣ←→\r\xa0\u3000、，。「」！？；：<>")
    for p in punctuation:
        content_strings = content_strings.replace(p, " ")
    return(content_strings)
# =============================================================================
# Remove stopwords
# =============================================================================
def remove_stopwords(content_strings2, stopwords):
    for w in stopwords:
        content_strings2 = content_strings2.replace(w," ")
    return content_strings2
# =============================================================================
# n-gram
# =============================================================================
def ngram(string, a):
    tmp = []
    for n in range (2,a+1):
        for i in range(0,len(string)-n+1):
            term = string[i:i+n]
            if " " not in term:
                tmp.append(term)
    return tmp
# =============================================================================
# Read Documents
# =============================================================================

raw = pd.read_excel('hw1_text.xlsx')

# =============================================================================
# preprocessing all documents(removing puntuations and stopwords)
# ============================================================================= 

all_title = []
all_content = []
for i in range(raw.shape[0]):
    if(i%100==0):
        print("i="+str(i)+"\n")
    current_title = raw['標題'][i]
    current_content = raw['內容'][i]
    tmp_title = remove_puntuations(current_title)
    tmp_content = remove_puntuations(current_content)
    all_title.append(remove_stopwords(tmp_title,stopwords))
    all_content.append(remove_stopwords(tmp_content,stopwords))

num_all_doc = len(all_title)

# =============================================================================
# n-gram processing
# =============================================================================

stem_all_word = []
stem_all_title = []

for i in range(len(all_title)):
    stem_all_word.append(ngram(all_title[i],6)+ngram(all_content[i],6))
    stem_all_title.append(ngram(all_title[i],6)) 

# =============================================================================
# count all DF&TF for keywords
# =============================================================================
#count document frequency - all
corp_all_counter = Counter( word for vocabulary in stem_all_word for word in list(set(vocabulary)))

#count term frequency - all
term_all_counter = Counter(word for vocabulary in stem_all_word for word in list(vocabulary))

# =============================================================================
# Find catagory ##change catagory from here
# =============================================================================
stem_word = []
stem_title = []
num_doc = 0
category ="銀行"

for i in range (len(stem_all_word)):
    if(i%100==0):
        print(str(i))
    if category in stem_all_word[i]:
        stem_word.append(stem_all_word[i])
        stem_title.append(stem_all_title[i])
        num_doc+=1

# =============================================================================
# count category DF&TF
# =============================================================================
#count document frequency
corp_counter = Counter(word for vocabulary in stem_word for word in list(set(vocabulary)))
#sorting from high frequency to low frequency
df_vector = list(list(corp_counter.most_common()))
df_vector.sort(key=lambda tup: tup[0])

#count term frequency
term_counter = Counter(word for vocabulary in stem_word for word in list(vocabulary))
#sorting from high frequency to low frequency
tf_vector = list(list(term_counter.most_common()))
tf_vector.sort(key=lambda tup: tup[0])

print("END count category DF&TF")

# =============================================================================
# find catagory words' all DF&TF
# =============================================================================
alldf_vector = []
alltf_vector = [] 
for term, _ in df_vector:
    alldf_vector.append([term,corp_all_counter[term]])
    alltf_vector.append([term,term_all_counter[term]])

print("END find catagory words' all DF&TF")

# =============================================================================
# count TF-IDF
# =============================================================================
#inverse document frequency(/num_doc) vector 
idf_vector = np.zeros([len(df_vector)], dtype=np.float32)
for i in range(len(df_vector)):
    doc_freq = df_vector[i][1]
    idf_vector[i] =  -np.log10(doc_freq/num_doc)
    
#TF-IDF vector
tf_vector_onlynum = np.zeros([len(tf_vector)],dtype = np.int)
for i in range(len(tf_vector)):
    tf_vector_onlynum[i] += tf_vector[i][1]

tfidf_vector = (1+np.log10(tf_vector_onlynum))*idf_vector

print("END count TF-IDF")
# =============================================================================
# count all TF-IDF
# =============================================================================
#inverse document frequency(/num_all_doc) vector 
idf_all_vector = np.zeros([len(alldf_vector)], dtype=np.float32)
for i in range(len(alldf_vector)):
    alldoc_freq = alldf_vector[i][1]
    idf_all_vector[i] =  -np.log10(alldoc_freq/num_all_doc)

#TF-IDF vector
alltf_vector_onlynum = np.zeros([len(alltf_vector)],dtype = np.int)
for i in range(len(alltf_vector)):
    alltf_vector_onlynum[i] += alltf_vector[i][1]
    
tfidf_all_vector = (1+np.log10(alltf_vector_onlynum))*idf_all_vector

print("END count all TF-IDF")
# =============================================================================
# count Expected value(TF&DF)
# =============================================================================
#TF_Expected value
tfe_vector = alltf_vector_onlynum*num_doc
for i in range(len(tfe_vector)):
    tfe_vector[i] = float(tfe_vector[i])/num_all_doc
    if(tfe_vector[i]<1):
        tfe_vector[i]=1

#DF_Expected value 
alldf_vector_onlynum = np.zeros([len(alldf_vector)],dtype = np.int)
for i in range(len(alldf_vector)):
    alldf_vector_onlynum[i] += alldf_vector[i][1]

dfe_vector = alldf_vector_onlynum*num_doc
for i in range(len(dfe_vector)):
    dfe_vector[i] = float(dfe_vector[i])/num_all_doc
    if(dfe_vector[i]<1):
        dfe_vector[i]=1
# =============================================================================
# count chi-square(TF&DF)
# =============================================================================
#TF_chi-square
tfc_vector = tf_vector_onlynum-tfe_vector
tfc_vector = tfc_vector*tfc_vector/ tfe_vector

for i in range(len(tfc_vector)):
    if(tf_vector_onlynum[i]-tfe_vector[i]<0):
        tfc_vector[i] *= -1 
    
#DF_chi-square
df_vector_onlynum = np.zeros([len(df_vector)],dtype = np.int)
for i in range(len(df_vector)):
    df_vector_onlynum[i] += df_vector[i][1]

dfc_vector = df_vector_onlynum - dfe_vector
dfc_vector = np.multiply(dfc_vector,dfc_vector)/dfe_vector
for i in range(len(dfc_vector)):
    if(df_vector_onlynum[i]-dfe_vector[i]<0):
        dfc_vector[i] *= -1

# =============================================================================
# count MI
# =============================================================================
#DF_MI  
tmp_vector = alldf_vector_onlynum*num_doc
MI_vector = df_vector_onlynum/tmp_vector
for i in range(len(MI_vector)):
   MI_vector[i] =  np.log10(MI_vector[i])

# =============================================================================
# count Lift
# =============================================================================
tmp_vector2 = alldf_vector_onlynum/num_all_doc
LIFT_vector = df_vector_onlynum/num_doc
LIFT_vector /= tmp_vector2

# =============================================================================
# output to csv
# =============================================================================
count = 1
f = codecs.open('銀行.csv','w','utf_8_sig') 
w = csv.writer(f)
w.writerow([num_doc,"docs"])
w.writerow(["編號","詞","TF","DF","TF-IDF","全部TF","全部DF","全部TF-IDF","TF期望值","DF期望值","TF卡方值(保留正負號)","DF卡方值(保留正負號)","MI(用DF)","Lift(用DF)"]) 
for i in range(len(tf_vector)):
    if(tf_vector[i][1]>=50 and df_vector[i][1]>=6):
        content = [str(count),tf_vector[i][0],str(tf_vector[i][1]),str(df_vector[i][1]),str(round(tfidf_vector[i],6)),str(alltf_vector[i][1]),str(alldf_vector[i][1]),str(round(tfidf_all_vector[i],6)),str(round(tfe_vector[i],0)),str(round(dfe_vector[i],0)),str(round(tfc_vector[i])),str(round(dfc_vector[i])),str(round(MI_vector[i],6)),str(round(LIFT_vector[i],6))]
        w.writerow(content)
        count+=1
print("Total words: "+str(count))
