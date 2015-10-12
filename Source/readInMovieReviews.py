import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import movie_reviews
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pickle

from nltk.classify.scikitlearn import SklearnClassifier



# --- PREPROCESSING THE DATA ---

documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append( (list(movie_reviews.words(fileid)), category) )

random.shuffle(documents)


# get a list of words in lower case
all_words = []
for word in movie_reviews.words():
    all_words.append(word.lower())

# filter out the stop words
stop_words = set(stopwords.words("english"))

filtered_words = [word for word in all_words if not word in stop_words]

# -- stem words --
#ps = PorterStemmer()
#stemmed_filtered_words = [ps.stem(word) for word in filtered_words]

# -- lemmatize --
#lemmatizer = WordNetLemmatizer()
#lemmatized_filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print all_words [:100]

# get the word frequency
all_words = nltk.FreqDist(all_words) # ordered words according to the amount of its appearances
#print(all_words.most_common(15)) # show the 15 most common words
#print(all_words["stupid"]) #shows how many times this word appears

word_features = list(all_words.keys()) [:3000] #only look at 3000 most common words

def find_features(document):
    words = set(document) # first part of the tuple
    features = {} # empty dictionary
    for word in word_features:
        features[word] = (word in words)
    
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))
featuresets = [(find_features(rev), category) for (rev, category) in documents]

training_set = featuresets[:1900] #take the first 1900 words as a training set

testing_set = featuresets[1900:] #take the other words as the test set






# --- NAIVE BAYES ALGORITHM WITH NLTK ---

classifier = nltk.NaiveBayesClassifier.train(training_set)
#print("Naive Bayes Algo acczraxe: ", (nltk.classify.accuracy(classifier, testing_set))*100)

# get most informative features for both categories
#classifier.show_most_informative_features(15)


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

print("MNB_classifier acczraxe: ", (nltk.classify.accuracy(MNB_classifier, training_set))*100)

# GaussianNB
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)

#print("GaussianNB acczraxe: ", (nltk.classify.accuracy(GaussianNB_classifier, testing_set))*100)

# BernoulliNB
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)

print("BernoulliNB acczraxe: ", (nltk.classify.accuracy(BernoulliNB_classifier, training_set))*100)




from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# LogisticRegression
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)

print("LogisticRegression acczraxe: ", (nltk.classify.accuracy(LogisticRegression_classifier, training_set))*100)

# SGDClassifier
SGD_classifier = SklearnClassifier(SGDClassifier())
SGD_classifier.train(training_set)

print("SGDClassifier acczraxe: ", (nltk.classify.accuracy(SGD_classifier, training_set))*100)


# SVC
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)

print("SVC acczraxe: ", (nltk.classify.accuracy(SVC_classifier, training_set))*100)

# LinearSVC
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)

print("LinearSVC acczraxe: ", (nltk.classify.accuracy(LinearSVC_classifier, training_set))*100)

# NuSVC
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)

print("NuSVC_classifier acczraxe: ", (nltk.classify.accuracy(NuSVC_classifier, training_set))*100)


# --- COMBINING ALGOS ---
#from nltk.classify import ClassifierI
#from statistics import mode

#class VoteClassifier(ClassifierI):
#   def _init_(self, *classifiers):
#       self._classifiers = classifiers
        
        #   def classify(self, features):
        #votes = []
        #for c in self._classifiers:
        #   v = c.classify(features)
        #   votes.append(v)
        #return mode(votes)
        
        #def confidence(self, features):
        #votes = []
        #for c in self._classifiers:
        #   v = c.classify(features)
        #   votes.append(v)
        #choice_votes = votes_count(mode(votes))
        #conf = choice_votes / len(votes)
#return conf


#voted_classifier = VoteClassifier(classifier,MNB_classifier)
                                  #,BernoulliNB_classifier, LogisticRegression_classifier,SVC_classifier,LinearSVC_classifier,NuSVC_classifier)

#print("voted_classifier acczraxe: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)
