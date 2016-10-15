#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate LDA Model
python TM_lda.py -f directory_of_corpus -k number_of_topics -alpha hyperparameter -beta hyperparameter -i number_of_iteration
"""
import numpy as np
import sys


modelfile = "./model/final"  # the location where to store the model file
#including the following files: final.twords, final.theta, final.phi


class LDA:
    def __init__(self, K, alpha, beta, docs, V):  
        self.K = K  # the number of topics
        self.alpha = alpha  # hyperparameter alpha: doc-topic distribution
        self.beta = beta   # hyperparameter beta: topic-terms distribution
        self.docs = docs
        self.V = V  # the total number of terms
        self.M = len(self.docs)  # the number of docs
        
        self.z_m_n = []  # the topic of nth words of mth doc
        self.n_m_k = np.zeros((self.M, K)) + alpha  #the number of words whose topic is k in every doc m，M*K matrix，M docs K topics，initialize every element to (0+alpha)
        self.n_k_t = np.zeros((K, V)) + beta  # the number of whose terms is t in every topic k, K*V matrix,K topics V terms, initialize every element to (0+beta)
        self.n_k = np.zeros(K) + V * beta   # the number of words of every topic k, a K-dimensional vector, initialize every element to (V*beta)
        self.n_m = np.zeros(self.M) + K * alpha  # the number of topics of every doc m,a M-dimensional vector,initialize every element to (K*alpha)

        self.theta = np.zeros((self.M, K))  # doc-topic distribution, M*K
        self.phi = np.zeros((K, V))  #topic-terms distribution, K*V

        self.N = 0
        for m, doc in enumerate(docs):
            self.N += len(doc)  # calculate the total number of words in all the docs
            z_n = []  # use to store the topics of every term, for every doc there is a z_n
            for t in doc:
                # assign an initial topic to term t
                p_z = self.n_k_t[:, t] * self.n_m_k[m] / self.n_k  #（tth colum）K*1 matrix made up by all the topics whose term is t *（mth row）1*k matrix made up by all the topics of mth doc, then get K*K，every element is alpha*beta
                z = np.random.multinomial(1, p_z / p_z.sum()).argmax()  # multinomial distribution sampling，and get the maximun
                z_n.append(z)
                self.n_m_k[m, z] += 1
                self.n_k_t[z, t] += 1
                self.n_k[z] += 1
                self.n_m[m] += 1  
            self.z_m_n.append(np.array(z_n))  # append the list which is made by every term's topic in this doc to z_m_n

    def inference(self):
        #learning once iteration
        for m, doc in enumerate(self.docs):
            for n, t in enumerate(doc):

                z = self.z_m_n[m][n]   #z is the topic of nth term of mth doc
                self.n_m_k[m, z] -= 1
                self.n_k_t[z, t] -= 1
                self.n_k[z] -= 1

                # sample new topic for term t
                p_z = self.n_k_t[:, t] * self.n_m_k[m] / self.n_k   
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
              
                # set new_z as the new topic and add count
                self.z_m_n[m][n] = new_z                
                self.n_m_k[m, new_z] += 1
                self.n_k_t[new_z, t] += 1
                self.n_k[new_z] += 1
                
    def worddist(self):
        # get topic-terms distribution
        return self.n_k_t / self.n_k[:, np.newaxis]

    # perplexity=exp^{ - (∑log(p(w))) / (N) }
    # p(w) is the probability of appearance of every word in test set, in LDA model, p(w)=∑z p(z|d)*p(w|z)
    # N is the length of test set
    def perplexity(self, docs=None): 
        if docs == None: docs = self.docs
        phi = self.worddist()
        log_per = 0
        N = 0
        Kalpha = self.K * self.alpha
        for m, doc in enumerate(docs):
            self.theta[m] = self.n_m_k[m] / (len(self.docs[m]) + Kalpha)
            for w in doc:
                log_per -= np.log(np.inner(phi[:, w], self.theta[m]))
            N += len(doc)
        return np.exp(log_per / N), self.theta


def lda_learning(lda, iteration, voca):
    pre_perp, theta = lda.perplexity()
    for i in range(iteration):
        lda.inference()
        perp, t = lda.perplexity()  
        if pre_perp:
            if pre_perp < perp: # if the after-sampled model is better than before-sampled model
                output_word_topic_dist(lda, voca)
                pre_perp = None
            else:
                pre_perp = perp
    phi = output_word_topic_dist(lda, voca)
    save_model(lda, theta, phi)


def output_word_topic_dist(lda, voca):  # output topic-terms distribution
    zcount = np.zeros(lda.K, dtype=int)
    wordcount = [dict() for k in xrange(lda.K)]
    for xlist, zlist in zip(lda.docs, lda.z_m_n):
        for x, z in zip(xlist, zlist):
            zcount[z] += 1
            if x in wordcount[z]:
                wordcount[z][x] += 1
            else:
                wordcount[z][x] = 1

    phi = lda.worddist()

    with open(modelfile+'.twords', 'w') as ftwords:   # save topic-word
        for k in xrange(lda.K):
            ftwords.write("topic: %d (%d words)" % (k, zcount[k]))
            ftwords.write('\n')
            for w in np.argsort(-phi[k])[:20]:
                ftwords.write("\t %s: %f (%d)" % (voca[w], phi[k, w], wordcount[k].get(w, 0)))
                ftwords.write('\n')
       
    return phi


def save_model(lda, theta, phi):  # save theta and phi
    with open(modelfile+'.theta', 'w') as ftheta:  # save theta
        for x in xrange(lda.M):  # M docs
            for y in xrange(lda.K):  # K topics
                ftheta.write(str(theta[x, y]) + ' ')
            ftheta.write('\n')

    with open(modelfile+'.phi', 'w') as fphi:  # save phi
        for x in xrange(lda.K):
            for y in xrange(lda.V):
                fphi.write(str(phi[x, y]) + ' ')
            fphi.write('\n')


if __name__ == "__main__":
    import optparse
    import TM_vocabulary

    parser = optparse.OptionParser()
    parser.add_option("-f", dest="parent_dir", help="parent_dir")
    parser.add_option("-ud", dest="userdict_dir", help="userdict_dir")
    parser.add_option("-alpha", dest="alpha", type="float", help="parameter alpha", default=0.5)
    parser.add_option("-beta", dest="beta", type="float", help="parameter beta", default=0.5)
    parser.add_option("-k", dest="K", type="int", help="number of topics", default=10)
    parser.add_option("-i", dest="iteration", type="int", help="iteration count", default=100)
    (options, args) = parser.parse_args()
    if not options.parent_dir:
        parser.error("need corpus parent_dir(-f)")

    if options.parent_dir:
        corpus = TM_vocabulary.load_file(options.parent_dir, options.userdict_dir)
        voca = TM_vocabulary.Vocabulary()
        docs = [voca.doc_to_ids(doc) for doc in corpus]
        lda = LDA(options.K, options.alpha, options.beta, docs, voca.size())
        lda_learning(lda, options.iteration, voca)