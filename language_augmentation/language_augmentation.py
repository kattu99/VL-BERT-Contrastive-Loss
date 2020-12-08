import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import wordnet 
from nltk.corpus import stopwords  
import gensim.downloader as api
import time
import os
import string

os.environ["GENSIM_DATA_DIR"] = "/home/proj/VL-BERT15/language_augmentation/"

class LanguageAugmentation: 

    def __init__(self, vocab): 
        self.word_set = set()
        self.stop_words = set(stopwords.words('english'))  
        for tup in vocab:
            self.word_set.add(tup)
        self.model = api.load("glove-twitter-25")
        self.vocab = self.model.vocab

    def augment_sentence(self, old_tokens):
        tokens = old_tokens.copy()
        for i in range(len(tokens)):
            if tokens[i] in string.punctuation:
                continue
            if tokens[i] not in self.stop_words:
                synonyms = self.get_synonyms(tokens[i])
                synonyms.sort(key=lambda x: (x[1], -x[2]))
                if len(synonyms) > 0: 
                    for synonym in synonyms:
                        if synonym[0] != tokens[i]:
                            tokens[i] = synonym[0]
                            break
        return tokens 

    def get_synonyms(self,word):
        synonyms = []
        syns = wordnet.synsets(word)
        second_syns = []
        if word in self.vocab:
            second_syns = self.model.most_similar(word)
        start = time.time()
        for syn in syns: 
            for l in syn.lemmas():
                # check the model entry 
                if l.name() in self.vocab and word in self.vocab:
                    # check the VL-BERT word set
                    if l.name() in self.word_set: 
                        synonyms.append((l.name(), 1, self.model.similarity(l.name(), word)))
                else: 
                    if l.name() in self.word_set: 
                        synonyms.append((l.name(), 1, 0))
        
        for second_sub in second_syns:
            second = second_sub[0]
            if second in self.word_set: 
                synonyms.append((second, 1, second_sub[1]))

        end = time.time()
        return synonyms

