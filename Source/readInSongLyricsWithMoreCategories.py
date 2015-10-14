import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

import random
import re
import csv
import pickle



def getMood(subMood):
    
    
    posMood = ["amiable-good-natured", "boisterous", "bright", "campy", "cheerful",
               "effervescent", "euphoric", "exuberant", "fun", "giddy", "gleeful",
               "happy", "joyous", "laid-back-mallow", "lazy", "light", "pastoral",
               "precious", "reverent", "romantic", "sensual", "silly", "smooth",
               "sparkling", "spicy", "springlike", "stylish", "sugary", "summery",
               "sweet", "thrilling", "warm"]
        
    negMood = ["aggressive", "angry", "anguished-distraught", "autumnal", "bleak",
                "brittle", "brooding", "circular", "cold", "complex", "dark", "detached",
                "difficult", "druggy", "eccentric", "eerie", "enigmatic", "epic",
                "fierce", "fractured", "gloomy", "harsh", "hostile", "hungry", "insular",
                "knotty", "manic", "meandering", "naive", "outraged", "outrageous",
                "paranoid", "rustic", "self-conscious", "spacey", "sparse", "spooky",
                "suspenseful", "thuggish", "uncompromising", "volatile", "wintry"]
    subMoods = subMood.split(',')
    
    posCount = 0
    negCount = 0
    
    for mood in subMoods:
        if mood in posMood:
            posCount+=1
        elif mood in negMood:
            negCount+=1

    if posCount > negCount:
        return 'pos'

    if negCount > posCount:
        return 'neg'




def preproccessLyrics(lyrics):

    #get words without special characters
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]',r'', re.sub(r'[?|$|.|!]',r'',lyrics)))

    stop_words = set(stopwords.words("english"))
    filtered_word_tokens = [word.lower() for word in word_tokens if not word in stop_words]
    #filtered_word_tokens = [word.lower() for word in word_tokens]


    lemmatizer = WordNetLemmatizer()
    lemmatized_filtered_words = [lemmatizer.lemmatize(word) for word in filtered_word_tokens]
    #ps = PorterStemmer()
    #stemmed_filtered_word_tokens = [ps.stem(word) for word in filtered_word_tokens]

    preproccessedLyrics = ""

    for word in lemmatized_filtered_words:
        preproccessedLyrics += " " + word

    return preproccessedLyrics


documents = []
all_words = ""


with open('songsClusteredMoods.csv', 'rU') as f:
    reader = csv.reader(f, delimiter=';', dialect=csv.excel_tab)
    for line in reader:
        mood = getMood(line[0])
        documents.append( (preproccessLyrics(line[1]), mood) )
        all_words += " " + preproccessLyrics(line[1])

random.shuffle(documents)

#save documents
save_documents = open("documents.pickle", "w") # use mode "rw" in windows
pickle.dump(documents, save_documents)
save_documents.close()

all_words_list = all_words.split()

#save dictionary
save_dictionary = open("dictionary.pickle", "w") # use mode "rw" in windows
pickle.dump(all_words_list, save_dictionary)
save_dictionary.close()