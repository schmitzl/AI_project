import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import csv

all_lyrics = '';

with open('songs.csv', 'rb') as f:
    reader = csv.reader(f, delimiter=';')
    for line in reader:
        all_lyrics += line[1]


#print all_lyrics

word_tokens = word_tokenize(all_lyrics)

#print word_tokens

stop_words = set(stopwords.words("english"))

filtered_word_tokens = [word for word in word_tokens if not word in stop_words]

#print filtered_word_tokens

# -- stem words --
ps = PorterStemmer()
#stemmed_filtered_word_tokens = [ps.stem(word) for word in filtered_word_tokens]

#print stemmed_filtered_word_tokens

#print stemmed_filtered_word_tokens.size()


# -- lemmatize --
lemmatizer = WordNetLemmatizer()
#lemmatized_filtered_word_tokens = [lemmatizer.lemmatize(word) for word in filtered_word_tokens]
