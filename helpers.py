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
        return data
        
    
    for file_name in os.listdir(data_set_path):
        file_path = os.path.join(data_set_path, file_name)
        file = open(file_path, 'r')

        # read raw text
        text = file.read()

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

def get_doc_vecs_for_data(data, word_embeddings, dimensions, word_weights = None):
    doc_vecs = np.empty([len(data), dimensions])

    for idx,review in enumerate(data):
        doc_vec = np.zeros([1,dimensions])
        for word in review:
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
    
    # shuffle the dRandomForestClassifierata and the labels
    idx = np.random.permutation(data.shape[0])
    
    data = data[idx]
    labels = labels[idx]
    
    # get kfolds
    # for each classifier, test it on each of the k folds
    clf_score_dict = {}
    for clf in clfs:
        scores = cross_val_score(clf, data, y=labels, cv=n_splits)
        avg_score = sum(scores)/n_splits
        clf_score_dict[clf] = avg_score
        
    return clf_score_dict