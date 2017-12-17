#coding=utf8
'''
    an util encapsules LDA from the package gensim
'''
from nltk.tokenize import RegexpTokenizer
from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim
import re


class LDA(object):

    def __init__(self, K=10, doc_set=[], model_filename='', load_from_file=False):
        if load_from_file:
            self.ldamodel=gensim.models.ldamodel.LdaModel.load(model_filename)
            self.term2id = {v:k for k,v in self.ldamodel.id2word.items()}
            print 'load model from %s...' % model_filename
        else:
            print 'initials: topic_num=%s, num_docs=%s' % (K, len(doc_set))
            self.K = K
            self.texts = doc_set
            #self.filename = filename

    def load_texts(self):
        lines = open(self.filename, 'r').readlines()
        self.texts = [l.strip() for l in lines]
        print 'load %s docs...' % len(self.texts)

    def check_existence_doc_term(self, terms):
        '''
            Given a document by a list of terms, check whether all the terms are in the corpus of the model
        '''
        res = []
        for ind, t in enumerate(terms):
            if t not in self.term2id:
                res.append((ind,t))
        return res

    def get_document_topics(self, terms):
        bow = {}
        for t in terms:
            if t in self.term2id:
                bow[self.term2id[t]] = bow.get(self.term2id[t], 0) + 1
        topic_ids = self.ldamodel.get_document_topics(bow.items())
        return sorted(topic_ids, key=lambda d:d[1], reverse=True)

    def train(self):

        # turn our tokenized documents into a id <-> term dictionary
        self.dictionary = corpora.Dictionary(self.texts)
        self.term2id = {v:k for k,v in self.dictionary.items()}

        # convert tokenized documents into a document-term matrix
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]

        # generate LDA model
        self.ldamodel = gensim.models.ldamodel.LdaModel(self.corpus, num_topics=self.K, id2word = self.dictionary)

        topics = self.ldamodel.print_topics(num_topics=-1)

        for t in topics:
            print t

    def print_topics(self, num_topics, num_words=10):
        res = self.ldamodel.print_topics(num_topics, num_words)

        print res

        return res

    def save(self, filename):
        self.ldamodel.save(filename)
        print 'model saved in ', filename

def preprocessing(texts):

    #remove numbers
    texts = [re.sub(r'\d+', '', t) for t in texts]

    tokenizer = RegexpTokenizer(r'\w+')

    # clean and tokenize document string
    tokenized_texts = [tokenizer.tokenize(t.lower()) for t in texts]


    # create English stop words list
    en_stop = get_stop_words('en')

    # Create p_stemmer of class PorterStemmer
    p_stemmer = PorterStemmer()

    # remove stop words from tokens
    pos_texts = []
    for t in tokenized_texts:
        stopped_tokens = [i for i in t if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        pos_texts.append(stemmed_tokens)

    return pos_texts

if __name__ == '__main__':

    # create sample documents
    doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
    doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
    doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
    doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
    doc_e = "Health professionals say that brocolli is good for your health."


    # compile sample documents into a list
    test_doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]
    texts = preprocessing(test_doc_set)

    lda = LDA(doc_set=texts)
    lda.train()
    lda.print_topics(num_topics=2, num_words=4)
