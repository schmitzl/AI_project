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
    
    cluster1 = ["passionate", "rousing", "confident", "boisterous", "rowdy"]
    
    cluster2 = ["rollicking", "cheerful", "fun", "sweet", "amiable-good-natured"]
    
    cluster3 = ["literate", "poignant", "wistful", "bittersweet", "autumnal", "brooding"]
    
    cluster4 = ["humorous", "silly", "campy", "quirky", "whimsical", "witty", "wry"]
    
    cluster5 = ["aggressive", "fiery", "tense-anxious", "intense", "volatile", "visceral"]
    

    subMoods = subMood.split(',')
    
    clusterCount = [0,0,0,0,0]

    for mood in subMoods:
        if mood in cluster1:
            clusterCount[0]+=1
        elif mood in cluster2:
            clusterCount[1]+=1
        elif mood in cluster3:
            clusterCount[2]+=1
        elif mood in cluster4:
            clusterCount[3]+=1
        elif mood in cluster5:
            clusterCount[4]+=1

    if sum(clusterCount) is 0:
        return None

    index = 0

    for i in range (0,5):
        if clusterCount[i] > clusterCount[index]:
            index = i

    print 'c' + str(index + 1)

    return 'c' + str(index + 1)


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