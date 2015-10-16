import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import pickle


lyrics_file = open("lyrics.txt", "r")
lyrics = lyrics_file.read()
lyrics_file.close()

lyrics = lyrics.lower()
lyrics = lyrics.replace('\n', ' ').replace('\r', '')
lyrics = ' '.join(lyrics.split())

word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]',r'', re.sub(r'[?|$|.|!]',r'',lyrics)))

stop_words = set(stopwords.words("english"))
filtered_word_tokens = [word.lower() for word in word_tokens if not word in stop_words]

lemmatizer = WordNetLemmatizer()
lemmatized_filtered_words = [lemmatizer.lemmatize(word) for word in filtered_word_tokens]


preproccessedLyrics = []
for word in lemmatized_filtered_words:
    preproccessedLyrics.append(word)


word_features_file = open("wordFeatures", "r")
word_features = pickle.load(word_features_file)
word_features_file.close()


features = {} # empty dictionary
for word in word_features:
    features[word] = (word in preproccessedLyrics)


classifier_c5_file = open("MNBayes_classifier_c5", "r")
classifier_c5 = pickle.load(classifier_c5_file)
classifier_c5_file.close()

classifier_pn_file = open("SVC_classifier_pn", "r")
classifier_pn = pickle.load(classifier_pn_file)
classifier_pn_file.close()

print classifier_pn.classify(features)

print "The lyrics are classified as " + classifier_pn.classify(features) + " and lie in cluster " + classifier_c5.classify(features) + "."