import nltk
import random
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import csv
import re
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

#Parser for cmd-line input
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#Variable n-gram (1-3)
parser.add_argument('-n', type=int, default=1, choices=[1, 2, 3], help='n-gram')
parser.add_argument('-l', metavar='<lyrics>', type=str, help='lyrics of the song')
#TODO if -l -> <lyrics> are required
#Set size of training and test set
parser.add_argument('-train', metavar='<size>', type=int, default=66, help='specifies the training_set size (in %)')
parser.add_argument('-test', metavar='<size>', type=int, default=33, help='specifies the test_set size (in %)')
#TODO need a better explanation of the "-w" value
parser.add_argument('-w', metavar='<size>', type=int, default=80, help='size of most common words for analysation')
parser.add_argument('-c', metavar='<size>', type=int, default=0, help='prints out the most <size> common words')
#TODO better help
parser.add_argument('-i', metavar='<size>', type=int, default=1, help='runs <size> times')
#TODO safe classifier with highest acccuracy (with path)
#Choose the labels
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-pn', action='store_true', help='uses pos/neg labels')
group.add_argument('-c5', action='store_true', help='uses 5 predefined clusters as labels')
args = parser.parse_args()

if args.train < 0 or args.test < 0 or args.w < 0 or args.c < 0:
    argparse.ArgumentParser.exit(parser, 'Only positive values for the arguments are allowed')
if args.train + args.test > 100:
    argparse.ArgumentParser.exit(parser, 'The training_set size and test_set cannot take more than 100%')
if args.w > 100:
    argparse.ArgumentParser.exit(parser, 'The word count cannot exceed 100%')

def getMoodPosNeg(subMood):
    
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

    if subMood in posMood:
        return "positive"
    elif subMood in negMood:
        return "negative"
    else:
        return None

def getMoodCluster(subMood):
    
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

    return 'c' + str(index + 1)

# --- PREPROCESSING THE DATA ---
def preproccessLyrics(lyrics):

    #get words without special characters
    word_tokens = word_tokenize(re.sub(r'[^a-zA-Z0-9 ]',r'', re.sub(r'[?|$|.|!]',r'',lyrics)))
    
    #  print word_tokens

    stop_words = set(stopwords.words("english"))
    filtered_word_tokens = [word.lower() for word in word_tokens if not word in stop_words]
    #filtered_word_tokens = [word.lower() for word in word_tokens]


    lemmatizer = WordNetLemmatizer()
    lemmatized_filtered_words = [lemmatizer.lemmatize(word) for word in filtered_word_tokens]
    #ps = PorterStemmer()
    #stemmed_filtered_word_tokens = [ps.stem(word) for word in filtered_word_tokens]

    preproccessedLyrics = []

    if args.n == 1:
        for word in lemmatized_filtered_words:
            preproccessedLyrics.append(word)
    elif args.n == 2:
        for x in range(0,len(lemmatized_filtered_words)-1):
            preproccessedLyrics.append(str(lemmatized_filtered_words[x] + " " + lemmatized_filtered_words[x+1]))
    elif args.n == 3:
        for x in range(0,len(lemmatized_filtered_words)-2):
            preproccessedLyrics.append(str(lemmatized_filtered_words[x] + " " + lemmatized_filtered_words[x+1]+ " " + lemmatized_filtered_words[x+2]))

    return preproccessedLyrics

documents = []
all_words = []

with open('songsClusteredMoods.csv', 'rU') as f:
    reader = csv.reader(f, delimiter=';', dialect=csv.excel_tab)
    for line in reader:
        if args.pn:
            mood = getMoodPosNeg(line[0])
        elif args.c5:
            mood = getMoodCluster(line[0])

        if mood is not None:
            documents.append( (list(preproccessLyrics(line[1])), mood) )
            all_words.extend(preproccessLyrics(line[1]))

# get the word frequency
all_words = nltk.FreqDist(all_words) # ordered words according to the amount of its appearances
if args.c > 0:
    print(all_words.most_common(args.c)) # show the <c> most common words

#size 17406 (7811 with c5, 8247 with pn)
word_features = list(all_words.keys())[:int(len(list(all_words.keys()))*(args.w*0.01))] #only look at <w> most common words

#Classifier (value, classifier)
n_bayes_values = []
mn_bayes_values = []
bernoulli_nb_values = []
logistic_regression_values = []
sgd_values = []
svc_values = []
linear_svc_values = []
for x in range(0, args.i):
    random.shuffle(documents)

    def find_features(document):
        words = set(document) # first part of the tuple
        features = {} # empty dictionary
        for word in word_features:
            features[word] = (word in words)
        
        return features
    
    #size 2649 (648 with c5, 710 with pn)
    featuresets = [(find_features(rev), category) for (rev, category) in documents]
    train_size = int(len(featuresets)*(args.train*0.01))
    training_set = featuresets[:train_size] #take the first <train> words as a training set
    test_size = int(len(featuresets)*(args.test*0.01))
    testing_set = featuresets[train_size:(train_size+test_size)] #take the other <test> words as the test set


    # --- NAIVE BAYES ALGORITHM WITH NLTK ---
    
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    n_bayes_values.append(((nltk.classify.accuracy(classifier, testing_set))*100, classifier))
    
    # get most informative features for both categories
    classifier.show_most_informative_features(15)
    
    # --- CLASSIFIER WITH SCIKITLEARN ---
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB
    
    # Naive Bayes
    MNB_classifier = SklearnClassifier(MultinomialNB())
    MNB_classifier.train(training_set)
    
    mn_bayes_values.append(((nltk.classify.accuracy(MNB_classifier, testing_set))*100, MNB_classifier))

    # BernoulliNB
    BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
    BernoulliNB_classifier.train(training_set)
    
    bernoulli_nb_values.append(((nltk.classify.accuracy(BernoulliNB_classifier, testing_set))*100, BernoulliNB_classifier))
    
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.svm import SVC, LinearSVC, NuSVC

    # LogisticRegression
    LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
    LogisticRegression_classifier.train(training_set)
    
    logistic_regression_values.append(((nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100, LogisticRegression_classifier))
    
    # SGDClassifier
    SGD_classifier = SklearnClassifier(SGDClassifier())
    SGD_classifier.train(training_set)
    
    sgd_values.append(((nltk.classify.accuracy(SGD_classifier, testing_set))*100, SGD_classifier))
    
    # SVC
    SVC_classifier = SklearnClassifier(SVC())
    SVC_classifier.train(training_set)
    
    svc_values.append(((nltk.classify.accuracy(SVC_classifier, testing_set))*100, SVC_classifier))
    
    # LinearSVC
    LinearSVC_classifier = SklearnClassifier(LinearSVC())
    LinearSVC_classifier.train(training_set)
    
    linear_svc_values.append(((nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100, LinearSVC_classifier))

def calcClassAcc(classSet):
    val = 0
    highestVal = 0
    highest = None
    for (acc, cl) in classSet:
        if acc > highestVal:
            highest = cl
        val += acc
    val /= len(classSet)
    return (val, highest)

print("Naive Bayes Accuracy: ", calcClassAcc(n_bayes_values)[0])
print("MNB_classifier accuracy: ", calcClassAcc(mn_bayes_values)[0])
print("BernoulliNB accuracy: ", calcClassAcc(bernoulli_nb_values)[0])
print("LogisticRegression accuracy: ", calcClassAcc(logistic_regression_values)[0])
print("SGD accuracy: ", calcClassAcc(sgd_values)[0])
print("SVC accuracy: ", calcClassAcc(svc_values)[0])
print("LinearSVC accuracy: ", calcClassAcc(linear_svc_values)[0])

#save classifier
save_classifier = open("SVC_classifier_pn", "w")
pickle.dump(calcClassAcc(svc_values)[1], save_classifier)
save_classifier.close()

#TODO safe best