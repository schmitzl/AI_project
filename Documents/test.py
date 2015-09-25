"""
        ------- README ------

How to perform feature extraction / get a dictionairy and create bag of words from 
it using scikit-learn python libraries.

Most stuff taken from http://scikit-learn.org/stable/modules/feature_extraction.html#the-bag-of-words-representation.  I also added some more commentary coz their explanation is rather poor imo.

Installation instructions here: http://scikit-learn.org/stable/install.html 

"""


# this is a comment

print "hello world"

# fooling around 

measurements = [{'city': 'Dubai', 'temperature': 33.},
{'city': 'London', 'temperature': 12.},
{'city': 'San Fransisco', 'temperature': 18.}]

print measurements

from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer()
f = vec.fit_transform(measurements).toarray()

[x,y] = f.shape

print x
print y

"""
output:

3
4

"""


# start here

corpus = ['This is the first document.',
'This is the second second document.',
'And the third one.',
'Is this the first document?']


from sklearn.feature_extraction.text import CountVectorizer


vectorizer = CountVectorizer(min_df=1)

vectorizer 
CountVectorizer(analyzer='word',binary=False,decode_error='strict',dtype='numpy.int64',encoding='utf-8',input='content',lowercase=True,max_df=1.0,max_features=None, min_df=1,ngram_range=(1, 1), preprocessor=None, stop_words=None,strip_accents=None, token_pattern='(?u)\\b\\w\\w+\\b',tokenizer=None, vocabulary=None)

# Extracts features and creates some dictionairy
# internaly (i need to find where it stores it) 
# it also creates a BoW representation
# for every input training document.

xx = vectorizer.fit_transform(corpus)

xx.toarray()  

print xx


# xx is an array where each row is basically a Bag of Words for every one of the above "documents". 
# The interesting (or maybe not so interesting) part, is that the dictionairy is formed internally
# from the 'corpus' . Notice that this fucntion automatically removes duplicate words. In this simple 'corpus' we 
# have 4 documents that contains 20 words in total. However, only 9 of these words are unique, thus
# we end up with an array 4x9 . The output is not really handy to be honest but ill see how to make it
# look better.

"""
	xx real output:

  (0, 8)	1
  (0, 3)	1
  (0, 6)	1
  (0, 2)	1
  (0, 1)	1
  (1, 8)	1
  (1, 3)	1
  (1, 6)	1
  (1, 1)	1
  (1, 5)	2
  (2, 6)	1
  (2, 0)	1
  (2, 7)	1
  (2, 4)	1
  (3, 8)	1
  (3, 3)	1
  (3, 6)	1
  (3, 2)	1
  (3, 1)	1

which is basically this: 

	xx output:

[0, 1, 1, 1, 0, 0, 1, 0, 1],    # bag 1
[0, 1, 0, 1, 0, 2, 1, 0, 1],    # bag 2
[1, 0, 0, 0, 1, 0, 1, 1, 0],    # bag 3
[0, 1, 1, 1, 0, 0, 1, 0, 1]     # bag 4

Once we get to this point with all of our trainig samples it would be cool to have another matrix 
with stored in string type the label of each of these bags (using the same index)

something like this:

xx=

[0, 1, 1, 1, 0, 0, 1, 0, 1],    # bag 1
[0, 1, 0, 1, 0, 2, 1, 0, 1],    # bag 2
[1, 0, 0, 0, 1, 0, 1, 1, 0],    # bag 3
[0, 1, 1, 1, 0, 0, 1, 0, 1]     # bag 4

yy=

['positive'], #refers to bag 1
['negative'], #refers to bag 2
['positive'], #refers to bag 3
['negative'], #refers to bag 4

"""


# Suppose that we have some test samples, documents/lyrics. 

# We need to get a bag of words for each of these samples too.

k=vectorizer.transform(['Something completely new.']).toarray()

l=vectorizer.transform(['This is Something completely new.']).toarray()

print k

print l 


"""
real outputs:

[[0 0 0 0 0 0 0 0 0]]
[[0 0 0 1 0 0 0 0 1]]

First is k , second is l. Notice that each of these new documents have been trasformed into a bag of words.

As expected 'k' is zeros as it doesnt contain any words from the dictionairy we learn earlier, so 
basically it contains 9 x zeros --> ( size(vocabulary) x occurences in each bin).

On the other hand 'l' contains two 1s (bin 3 & 8  [bin_range = 0-8]), ('This', 'is') found in the 
dictionairy, rest = zeros (no occurences in the dict).

Note:

For the classification part we are also going to need the labels of the test data in another matrix. 
Same fashion as the training set. 

We only going to use test labels in order to check if the predicted labels match to test labels. We CANNOT use them
to train the system , otherwise it would be considered cheat, and appart from that it would score 100% accuracy
so we dont want this.  

"""











