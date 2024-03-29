from nltk.corpus import wordnet
import os
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import numpy as np
from sklearn.model_selection import cross_val_score
from itertools import product
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from bs4 import BeautifulSoup as bs # library for removing html tags from text
import nltk # natural language tool kit: for text pre-processing
from nltk.corpus import stopwords # a set of common stopwords from nltk

import gensim
from bs4 import BeautifulSoup as bs
import nltk # natural language tool kit: for text pre-processing
import os # for listing directories
import numpy as np # no comment :P
from nltk.corpus import stopwords # a set of common stopwords from nltk
from gensim import models
from collections import namedtuple
import matplotlib.pyplot as plt


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def penn_to_wn(tag):
    return get_wordnet_pos(tag)

def get_bigrams(train_data):
    for docs in train_data: 
        bigram_finder = BigramCollocationFinder.from_words(docs)
        bigrams = bigram_finder.nbest(BigramAssocMeasures.chi_sq, 500)
        tokens = []
        for bigram_tuple in bigrams:
            x = "%s %s" % bigram_tuple
            tokens.append(x)
    print(tokens)


stop_words = set(stopwords.words('english'))
stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '``', "''", '...','the','for',"'s","'m"])
wnl = nltk.WordNetLemmatizer()
def read_data(data_set_path):
    data = []
    
    if any("all.txt" in s for s in os.listdir(data_set_path)):
        # read the reviews line by line
        all_data_file_path = os.path.join(data_set_path, 'all.txt')
        all_data_file = open(all_data_file_path, 'r')
        for line in all_data_file.readlines():
            vec = line.split(" ")
            vec = [item for item in vec if item != '\n']
            data.append(vec)
        all_data_file.close()
        return data
        
    stop_words = set(stopwords.words('english'))
    stop_words.update(['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}', '``', "''", '...','the','for',"'s","'m"])
    wnl = nltk.WordNetLemmatizer()
    
    for file_name in os.listdir(data_set_path):
        file_path = os.path.join(data_set_path, file_name)
        file = open(file_path, 'r')

        # read raw text
        text = file.read()

        file.close()

        # remove html tags
        text = bs(text, "html.parser").get_text()

        # tokenize the text
        tokens = nltk.word_tokenize(text)

        # get the part of speech tags for the tokens
        # This gives each token a tag from [NOUN, VERB, ADJ, ADV] which are
        # used by the lemmatizer to correctly lemmatize a word
        tagged_text = nltk.pos_tag(tokens)

        # lowercase, remove stop words, and lemmatize
        tokens = [wnl.lemmatize(tok.lower(), penn_to_wn(tag)) for (tok, tag) in tagged_text if tok.lower() not in stop_words and not tok.isdigit()]
        data.append(tokens)
    
    all_data_file_path = os.path.join(data_set_path, 'all.txt')
    all_data_file = open(all_data_file_path, 'w+')
    all_data_file.writelines(["%s\n" % " ".join(str(x) for x in item) for item in data])

    return data

def get_doc_vecs_for_data(data, word_embeddings, dimensions, word_weights = None, ignored_words=None):
    doc_vecs = np.empty([len(data), dimensions])

    for idx,review in enumerate(data):
        doc_vec = np.zeros([1,dimensions])
        for word in review:
            if ignored_words == None or word not in ignored_words:
                if word_weights != None:
                    if word in word_embeddings and word in word_weights:
                        doc_vec += word_weights[word]*word_embeddings[word]
                else:
                    if word in word_embeddings:
                        doc_vec += word_embeddings[word]
        doc_vecs[idx] = doc_vec/len(review)
    
    return doc_vecs

def load_glove_dict(size):
    assert size == 50 or size == 100 or size == 200 or size == 300
    
    glove_file_path = "glove.6B/glove.6B." + str(size) + "d.txt"
    
    glove_file = open(glove_file_path, 'r')
    
    glove_dict = {}
    for line in glove_file.readlines():
        tokens = line.split(" ")
        word = tokens[0]
        vec = [float(num) for num in tokens[1::]]
        glove_dict[word] = vec

    glove_file.close()
    return glove_dict


def combination_of_params(dictionary):
    return [dict(zip(dictionary, v)) for v in product(*dictionary.values())]


def get_clfs_for_combinations(classifier, param_vals_dict):
    clfs = []
    for comb in combination_of_params(param_vals_dict):
        clfs.append(classifier(**comb))
    return clfs


def cross_validate(data, labels, clfs, n_splits = 10):
    assert data.shape[0] == len(labels)
    
    # shuffle the data and the labels
    idx = np.random.permutation(data.shape[0])
    data = data[idx]
    labels = labels[idx]
    
    # get kfolds
    # for each classifier, test it on each of the k folds
    clf_score_dict = {}
    for clf in clfs:
        scores = cross_val_score(clf, data, y=labels, cv=n_splits)
        clf_score_dict[clf] = scores
        
    return clf_score_dict

def print_clf_scores(scores_dict, clf_params):
    for clf in scores_dict.keys():
        params = []
        for param in clf_params:
            params.append( (param, clf.get_params()[param]) )
        scores = scores_dict[clf]
        avg_score = sum(scores)/len(scores)
        print("clf with params: {}, score: {}".format(params, avg_score))

def plot_accuracies(accuracies_dict, xlabel=None, ylabel='Accuracy', title=None):
    """
    returns an error bar plot for the given parameters and accuracies
    """

    xs = None
    means = []
    errors = []
    params = list(accuracies_dict.keys())
    params = np.sort(params)
    
    if type(params[0]) is float:
        xs = params
    else:
        xs = list(range(1, len(params) + 1))

    for param in params:
        means.append(np.mean(accuracies_dict[param]))
        errors.append(np.std(accuracies_dict[param]))
    
    plt.subplot(111)
    plt.errorbar(xs, means, errors, linestyle='None', marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if type(params[0]) is not float:
        plt.xticks(xs, [str(param) for param in params])

    return plt.gcf()


def clf_to_accuracies_dict(clf_scores_dict, param):
    return dict([(clf.get_params()[param], clf_scores_dict[clf]) for clf in clf_scores_dict.keys()])
