import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

from nltk.classify.scikitlearn import SklearnClassifier



from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        choice_votes = votes.count(mode(votes))
        conf =  choice_votes / len(votes)
        return conf








documents_f = open("documents.pickle", "r")
documents = pickle.load(documents_f)
documents_f.close()

allwords_f = open("dictionary.pickle", "r")
all_words = pickle.load(allwords_f)
allwords_f.close()

#print documents

#print all_words


# get the word frequency
all_words = nltk.FreqDist(all_words) # ordered words according to the amount of its appearances
print(all_words.most_common(300)) # show the 15 most common words

#word_features = all_words[:3000] #only look at 6000 most common words
word_features = list(all_words.keys())[:100] #only look at 6000 most common words




def find_features(document):
    words = set(document) # first part of the tuple
    features = {} # empty dictionary
    for word in word_features:
        features[word] = (word in words)

    return features

featuresets = [(find_features(lyrics), category) for (lyrics, category) in documents]

training_set = featuresets[:1000] #take the first 5000 words as a training set

testing_set = featuresets[1000:] #take the other words as the test set






# --- NAIVE BAYES ALGORITHM WITH NLTK ---

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo acczraxe: ", (nltk.classify.accuracy(classifier, testing_set))*100)

# get most informative features for both categories
classifier.show_most_informative_features(1000)


# ATTENTION: in windows you have to use the modes "wb" and "rb" instead


# save classifier
#save_classifier = open("naivebayes.pickle", "w")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

# use stored classifier
#classifier_f = open("naivebayes.pickle", "r")
#classifier = pickle.load(classifier_f)
#classifier_f.close()
#classifier.show_most_informative_features(15)

# --- CLASSIFIER WITH SCIKITLEARN ---
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

# Naive Bayes
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)

print("MNB_classifier acczraxe: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GaussianNB
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)

#print("GaussianNB acczraxe: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

print("BernoulliNB accuracy: ", (nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100)




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

print("LogisticRegression accuracy: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDClassifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)

print("SGDClassifier accuracy: ", (nltk.classify.accuracy(SGD_classifier, testing_set))*100)


# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

print("SVC accuracy: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

print("LinearSVC accuracy: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

print("NuSVC_classifier accuracy: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)











voted_classifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier, LogisticRegression_classifier,SVC_classifier,LinearSVC_classifier,NuSVC_classifier)

print("voted_classifier accuracy: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

