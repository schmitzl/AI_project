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
    
    
    posMoods = ["airy", "ambitious", "amiable-good-natured", "animated", "athletic", "atmospheric",
                "boisterous", "bombastic", "brash", "bravado", "bright", "calm-peaceful",
                "carefree", "cartoonish", "cathartic", "celebratory", "cerebral", "cheerful",
                "circular", "complex", "confident", "cosmopolitan", "crunchy", "delicate",
                "devotional", "dignified-noble", "dreamy", "driving", "druggy", "earnest",
                "effervescent", "elaborate", "elegant", "energetic", "epic", "erotic",
                "ethereal", "euphoric", "exciting", "extroverted", "exuberant", "feverish",
                "flashy", "flowing", "freewheeling", "fun", "gentle", "gleeful", "graceful",
                "gutsy", "happy", "hedonistic", "heroic", "humorous", "hyper", "hypnotic",
                "indulgent", "innocent", "intimate", "joyous", "light", "literate", "lively",
                "lush", "lyrical", "mysterious", "nostalgic", "passionate", "pastoral", "perky",
                "philosophical", "playful", "powerful", "precious", "pulsing", "quirky", "rambunctious",
                "ramshackle", "reassuring-consoling", "refined", "relaxed", "resolute",
                "reverent", "raucous", "rollicking", "romantic", "rousing", "self-conscious",
                "sensual", "sexual", "sexy", "shimmering", "silly", "smooth", "soft-quiet",
                "soothing", "sophisticated", "sparkling", "spicy", "spiritual", "spontaneous",
                "sprawling", "springlike", "stately", "street-smart", "stylish", "sugary",
                "summery", "sweet", "tender", "theatrical", "thrilling", "triumphant",
                "tuneful", "uplifting", "virile", "visceral", "warm", "whimsical",
                "witty"]

    negMoods = ["acerbic", "aggressive", "angry", "angst-ridden", "anguished-distraught", "austere",
                "autumnal", "belligerent", "bitter", "bittersweet", "bleak", "brassy", "brittle",
                "brooding", "campy", "clinical", "cold", "confrontational", "cynical-sarcastic",
                "defiant", "desperate", "detached", "difficult", "dramatic", "earthy", "eccentric",
                "eerie", "enigmatic", "fierce", "fiery", "fractured", "giddy", "gloomy", "greasy",
                "gritty", "harsh", "hostile", "hungry", "insular", "intense", "introspective",
                "ironic", "irreverent", "kinetic", "knotty", "laid-back-mellow", "languid", "lazy", "lonely"
                "malevolent", "manic", "martial", "meandering", "melancholy", "menacing", "messy",
                "motoric", "mysterious", "narcotic", "nihilistic", "nocturnal", "ominous",
                "organic", "outraged", "outrageous", "paranoid", "plaintive", "poignant",
                "provocative", "rebellious", "reckless", "reflective", "regretful",
                "reserved", "restrained", "rowdy", "rustic", "sad", "sardonic", "savage",
                "scattered", "searching", "sentimental", "serious", "sleazy", "slick",
                "snide", "somber", "spacey", "sparse", "spooky", "striding", "suffocating",
                "swaggering", "tense-anxious", "thoughtful", "thuggish", "tragic", "trashy", "trippy",
                "turbulent", "uncompromising", "unsettling", "urgent", "vulgar", "weary",
                "wintry", "wistful", "wry", "yearning"]

    subMoods = subMood.split(',')
    
    posCount = 0
    negCount = 0
    
    for mood in subMoods:
        if mood in posMoods:
            posCount+=1
        else:
            negCount+=1

    if posCount > negCount:
        return 'pos'

    return 'neg'





def preproccessLyrics(lyrics):

    #get words without special characters
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]',r'', re.sub(r'[?|$|.|!]',r'',lyrics)))

    stop_words = set(stopwords.words("english"))
    filtered_word_tokens = [word.lower() for word in word_tokens if not word in stop_words]
    #filtered_word_tokens = [word.lower() for word in word_tokens]


    lemmatizer = WordNetLemmatizer()
    lemmatized_filtered_words = [lemmatizer.lemmatize(word) for word in filtered_words]
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
        documents.append( (preproccessLyrics(line[1]), getMood(line[0])) )
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