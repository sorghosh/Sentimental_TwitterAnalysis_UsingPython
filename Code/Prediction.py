import pickle
import nltk
import pandas as pd
import numpy as np
from statistics import mode

train_dataset = {}

def voterclassify(words):

    logistic_p = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\Logistic_Model.pickle","rb")
    logmodel = pickle.load(logistic_p)
    logistic_p.close()

    bernoulliNB_model_p = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\BernoulliNB_model.pickle","rb")
    bernoulliNB_model = pickle.load(bernoulliNB_model_p)
    bernoulliNB_model_p.close()

    multinomialNB_model_p = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\MultinomialNB_model.pickle","rb")
    multinomialNB_model = pickle.load(multinomialNB_model_p)
    multinomialNB_model_p.close()

    sgd_classifier_model_p = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\SGDClassifier_model.pickle","rb")
    sgd_classifier_model = pickle.load(sgd_classifier_model_p)
    sgd_classifier_model_p.close()

    randomforest_p = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\RandomForestClassifier_model.pickle","rb")
    randomforest = pickle.load(randomforest_p)
    randomforest_p.close()
    
    logmodel = logmodel.predict(words)
    bernoulliNB_model= bernoulliNB_model.predict(words)
    multinomialNB_model= multinomialNB_model.predict(words)
    sgd_classifier_model= sgd_classifier_model.predict(words)
    randomforest= randomforest.predict(words)

    best_vote_temp = []
    best_vote_temp.append(logmodel[0])
    best_vote_temp.append(bernoulliNB_model[0])
    best_vote_temp.append(multinomialNB_model[0])
    best_vote_temp.append(sgd_classifier_model[0])
    best_vote_temp.append(randomforest[0])
    best_vote = mode(best_vote_temp)
    confidence = float(best_vote_temp.count(best_vote))/float(len(best_vote_temp))
    
    return best_vote,confidence
        
def get_features(doc):
    words = nltk.word_tokenize(doc)
    for w in word_features:
        train_dataset[w] = (w in words)
    words = pd.DataFrame([w for w in train_dataset.values()]).T
    words = np.array(words)

    return words

word_features_f  = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\word_features.pickle")
word_features = pickle.load(word_features_f)
word_features_f.close()
#
#
#doc1 = "Welcome to Heathrow, Martin. Where in the world have you flown in from today?"
#words = get_features(doc1)
#voterclassify(words)
#
#
