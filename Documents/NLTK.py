import nltk

example_text = "Now I was sitting waiting wishing that you believed in superstitions then maybe you'd see the signs. But Lord knows that this world is cruel and I ain't the Lord, no I'm just a fool. Learning loving somebody don't make them love you"


# --------- WORD-TOKENIZER ---------
from nltk.tokenize import word_tokenize

word_tokens = word_tokenize(example_text)

# print(word_tokens)

# for word in word_tokens:
#     print(word)
    
    
    
    
# --------- STOP-WORDS ---------
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

# print stop_words

filtered_word_tokens = []

# long version

# for word in word_tokens:
#     if word not in stop_words:
#         filtered_word_tokens.append(word)

# short version

filtered_word_tokens = [word for word in word_tokens if not word in stop_words]

# print filtered_word_tokens




# --------- STEMMING ---------
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_filtered_word_tokens = [ps.stem(word) for word in filtered_word_tokens]

# print stemmed_filtered_word_tokens

# for word in stemmed_filtered_word_tokens:
#      print word


# --------- SPEECH TAGGING ---------
from nltk.corpus import state_union

# read in txt file
# train_text = state_union.raw("train.txt")

# categorizes words into verbs, adverbs, adjectives... not sure we need this - so no further code here
# in case we need it: watch video on https://www.youtube.com/watch?v=6j6M2MtEqi8


# --------- LEMMATIZING ---------
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


print lemmatizer.lemmatize("better")
print lemmatizer.lemmatize("better", pos="a")