import os
os.system('pip install nltk')
os.system('pip install stanza')
os.system('pip install emoji')
os.system('pip install -U sentence-transformers')

import pandas as pd
import numpy as np
try: from sentence_transformers import SentenceTransformer
except ModuleNotFoundError: pass
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
nltk.download('words')
words = set(nltk.corpus.words.words())
import stanza
stanza.download("en")

class Tweet():
    def __init__(self, text, text_clean, token, author_id):
        self.token = token
        self.text = text
        self.text_clean = text_clean
        self.author_id = author_id
        
        self.similiar_tweets = []
        self.similiar_authors = []

        self.sentiments = {}
        self.associations = []

class User():
    def __init__(self, author_id):
        self.author_id = author_id
        self.tweet_tokens = []
        self.similiar_authors = []
        self.author_edges = {}

#Removing Emojis
def remove_emojis(data):
    emoj = re.compile("["
                      u"\U0001F600-\U0001F64F"  # emoticons
                      u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                      u"\U0001F680-\U0001F6FF"  # transport & map symbols
                      u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                      u"\U00002500-\U00002BEF"  # chinese char
                      u"\U00002702-\U000027B0"
                      u"\U00002702-\U000027B0"
                      u"\U000024C2-\U0001F251"
                      u"\U0001f926-\U0001f937"
                      u"\U00010000-\U0010ffff"
                      u"\u2640-\u2642"
                      u"\u2600-\u2B55"
                      u"\u200d"
                      u"\u23cf"
                      u"\u23e9"
                      u"\u231a"
                      u"\ufe0f"  # dingbats
                      u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, '', str(data))

def cleaner(text):
    tweet = re.sub("@[A-Za-z0-9]+","",str(text)) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", str(text)) #Remove http links
    tweet = re.sub('[()!?]', ' ', str(text)) #removing punctuation
    tweet = re.sub('\[.*?\]',' ', str(text))
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(str(text))
                     if w.lower() in words or not w.isalpha())
    return text

def calculate_sentiments(text, stop_words, nlp):
    txt = text
    sentList = nltk.sent_tokenize(txt) # Splitting the text into sentences
    fcluster = []
    totalfeatureList = []
    finalcluster = []
    featureList = []
    categories = []
    dic = {}

    for line in sentList:
        # Remove links from line
        line = re.sub(r'http\S+|#', '', line)

        # Swap '-', ';', '*' with commas
        line = re.sub(':', '.', line)
        line = re.sub('\n|@', '', line)

        # Remove consecutive punctuation recursively
        r = re.compile(r'([.,/#!$%^&*;:{}=_`~()-])[.,/#!$%^&*;:{}=_`~()-]+')
        line = r.sub(r'\1', line)

        # Replace hashtags with association term
        line = re.sub('#', 'hashtag is ', line)

        try:
            newtaggedList = []
            txt_list = nltk.word_tokenize(line) # Splitting up into words
            taggedList = nltk.pos_tag(txt_list) # Doing Part-of-Speech Tagging to each word

            newwordList = []
            flag = 0
            for i in range(0,len(taggedList)-1):
                if(taggedList[i][1]=="NN" and taggedList[i+1][1]=="NN"): # If two consecutive words are Nouns then they are joined together
                    newwordList.append(taggedList[i][0]+taggedList[i+1][0])
                    flag=1
                else:
                    if(flag==1):
                        flag=0
                        continue
                    newwordList.append(taggedList[i][0])
                    if(i==len(taggedList)-2):
                        newwordList.append(taggedList[i+1][0])

            finaltxt = ' '.join(word for word in newwordList)
            new_txt_list = nltk.word_tokenize(finaltxt)
            wordsList = [w for w in new_txt_list if not w in stop_words]
            taggedList = nltk.pos_tag(wordsList)

            doc = nlp(finaltxt) # Object of Stanford NLP Pipeleine

            dep_node = []

            for dep_edge in doc.sentences[0].dependencies:
                dep_node.append([dep_edge[2].text, dep_edge[0].id, dep_edge[1]])

            for i in range(0, len(dep_node)):
                if (int(dep_node[i][1]) != 0):
                    dep_node[i][1] = newwordList[(int(dep_node[i][1]) - 1)]

            # featureList = []
            # categories = []
            for i in taggedList:
                if(i[1]=='JJ' or i[1]=='NN' or i[1]=='JJR' or i[1]=='NNS' or i[1]=='RB'):
                    featureList.append(list(i))
                    totalfeatureList.append(list(i)) # This list will store all the features for every sentence
                    categories.append(i[0])

            for i in featureList:
                filist = []
                for j in dep_node:
                    if((j[0]==i[0] or j[1]==i[0]) and (j[2] in ["nsubj", "acl:relcl", "obj", "dobj", "agent", "advmod", "amod", "neg", "prep_of", "acomp", "xcomp", "compound"])):
                        if(j[0]==i[0]):
                            filist.append(j[1])
                        else:
                            filist.append(j[0])
                fcluster.append([i[0], filist])

        except IndexError:
            print('IndexError:', line)
            return []

        except AttributeError:
            print('AttributeError')
            return []

    for i in featureList:
        dic[i[0]] = i[1]

    for i in fcluster:
        if(dic[i[0]]=="NN"):
            finalcluster.append(i)

    return finalcluster

def upload_to_output(path, bucket_name, folder_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(folder_name + '/' + path.split('/')[-1])
    blob.upload_from_filename(path)

from google.cloud import storage   
bucket_name = 'sw-airlines-data-hub'