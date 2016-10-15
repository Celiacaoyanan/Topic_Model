#!/usr/bin/env python
# -*- coding: utf-8 -*-

import jieba  # Chinese Word Segmentation
import os
import sys
reload(sys)
sys.setdefaultencoding( "utf-8" )


corpus = [] 


def load_file(parent_dir, userdict):  # read the doc and segment words
    jieba.load_userdict(userdict)
    fname_list = os.listdir(parent_dir)
    for fname in fname_list:
        with open(parent_dir + '//' + fname, 'r') as f:
            raw = f.read().decode('utf-8')
            word_list = list(jieba.cut(raw))
            corpus.append(word_list)
    '''
    # output the corpus after words segmentation
    f1 = open('wordlist.txt', 'w')
    for f in corpus:
        for w in f:
            w = w.strip()
            f1.write(w)
            f1.write(' ')
        f1.write('\n')
    '''
    return corpus
    

stopwords_list = open("stopwords.txt", "r")


def is_stopword(w):
    if w in stopwords_list:
        return True
     

class Vocabulary:
    def __init__(self):
        self.vocas = []         # id to word
        self.vocas_id = dict()  # word to id
        self.docfreq = []       # id to document frequency

    def term_to_id(self, term):  # get the id of this term
        if is_stopword(term):
            return None
        if term not in self.vocas_id:
            voca_id = len(self.vocas)  # if add this term to the end of vocas, so its id is also the last
            self.vocas_id[term] = voca_id
            self.vocas.append(term)  # add this term to the end of vocas
            self.docfreq.append(0) 
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):  # get all the ids of all the terms in this doc
        list = []
        words = dict()
        for term in doc:
            id = self.term_to_id(term)
            if id != None:
                list.append(id)
                if not words.has_key(id):
                    words[id] = 1
                    self.docfreq[id] += 1
        if "close" in dir(doc):
            doc.close()
        return list

    def size(self):
        return len(self.vocas)


