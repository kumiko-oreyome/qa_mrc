import jieba as jb
import json,os,pickle
from hanziconv import HanziConv
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity






def to_simplified_sentences(sentences):
    return [HanziConv.toSimplified(s) for s in sentences]

def read_txt_lines(path):
    lines = []
    with open(path,'r',encoding="utf-8") as f:
        for row in f:
            lines.append(row.rstrip('\n'))
    return lines
    
def read_json_utf8(path):
    with open(path,'r',encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def write_json_utf8(path,obj):
    with open(path,'w',encoding="utf-8") as f:
        json.dump(obj,f,ensure_ascii=False)
        
def generate_ngram(sentence,n):
    assert n > 0
    l = []
    for i in range(len(sentence)-n+1):
        l.append(sentence[i:i+n])
    return l

class Tokenizer():
    def __init__(self,stopword_path='./data/stopwords.txt'):
        self.stopword_list = []
        if stopword_path is not None:
            self.stopword_list =  read_txt_lines(stopword_path)
    def tokenize(self,s):
        return [w for w in jb.cut(s, cut_all=False) if w not in  self.stopword_list] 
    
class Tfidf():
    def __init__(self,sentences,tokenize_func,corpus_path=None):
        self.sentences = sentences
        self.tokenize_func = tokenize_func
        print('make corpus')
        if corpus_path is not None: 
            if not os.path.exists(corpus_path):
                self.corpus =  self.make_corpus(self.sentences)
                pickle.dump(self.corpus,open(corpus_path,'wb'))
            else:
                self.corpus = pickle.load(open(corpus_path,'rb'))
        else:
            self.corpus =  self.make_corpus(self.sentences)
        print('corpus complete')
        self.vectorizer = TfidfVectorizer(max_features=100000,min_df=2).fit(self.corpus)
    
    def make_corpus(self,sentences):
        wl =  list(map(lambda x:self.tokenize_func(x),sentences))
        return list(map(lambda x:" ".join(x),wl))

    def tfidf_transform(self,sentences):
        return  self.vectorizer.transform(self.make_corpus(sentences))
    
    def cosine_similarity(self,text1,text2):
        return  cosine_similarity(self.tfidf_transform([text1])[0],self.tfidf_transform([text2])[0])[0][0]

