import pandas as pd
import numpy  as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.cross_validation import ShuffleSplit
from sklearn import metrics
from sklearn.cross_validation import cross_val_score
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
#import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
import pickle


def ml_model(clf,x_train,x_test,y_train,y_test,x_features):

    clf.fit(x_train,y_train)    
    cv = ShuffleSplit(x_train.shape[0],n_iter = 20 , test_size = .30 , random_state = 0)

    cv_error    = []
    train_error = []
    test_error  = []
    
#    for train_cv , test_cv in cv:       
#        prd_cv  = clf.fit(x_train[train_cv],y_train[train_cv]).predict(x_train[test_cv])
#        cv_error.append(metrics.accuracy_score(y_train[test_cv],prd_cv)) 
#        
#        prd_train = clf.fit(x_train[train_cv],y_train[train_cv]).predict(x_train[train_cv])
#        train_error.append(metrics.accuracy_score(y_train[train_cv],prd_train))
#        
#        prd_test  = clf.fit(x_train[train_cv] , y_train[train_cv]).predict(x_test)
#        test_error.append(metrics.accuracy_score(y_test,prd_test))
                
    print "Model Report : "
    print "Training_Error:" , 1- metrics.accuracy_score(y_train,clf.fit(x_train,y_train).predict(x_train))
    print "Test_Error:" , 1- metrics.accuracy_score(y_test,clf.fit(x_train,y_train).predict(x_test))
    print "CV_Error:" , 1-  cross_val_score(clf,x_train,y_train,cv = 20).mean()
    

########plot the learning rate of the model 
#    x = range(0,20)
#    plt.plot(x,np.array(cv_error) , c = "red" ,linestyle = "--" ,label = "cv_score")
#    plt.plot(x,np.array(train_error) , c = "blue" ,linestyle = "--", label = "train_score")
#    plt.plot(x,np.array(test_error),c = "black", linestyle = "--",label ="test_score")
#    plt.legend(loc = "lower right")

#########plot the roc_curve and auc of the model
    y_pred_prob = clf.predict_proba(x_test)
    y_pred      = clf.predict(x_test) 
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_class = label_binarize(y_test,classes = [0,1,2])
    for i in range(len(y_class[0])):
        if i !=2:
            fpr[i],tpr[i],_ = roc_curve(y_class[:,i],y_pred_prob[:,i])
            roc_auc[i] = roc_auc_score(y_class[:,i],y_pred_prob[:,i])
#            plt.plot(fpr[i],tpr[i],label = "roc_auc_score for class {0} , area {1}".format(i,roc_auc[i]))

#    plt.plot([0,1],[0,1],"k--")
#    plt.xlabel("False Positive Rate")
#    plt.ylabel("True Positive Rate")
#    plt.legend(loc = "lower right",fontsize = 9)    
    print "roc_auc >>>>>>>>>" , roc_auc
    print "ConfusionMatrix >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
    print metrics.confusion_matrix(y_test,y_pred)
    
    return clf
    
training_dataset = pd.read_csv(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Training_Dataset.txt")
y = training_dataset["labels"]
training_dataset.pop("labels")
x_features = training_dataset.columns
y_features = "labels"
x_train,x_test,y_train,y_test = train_test_split(np.array(training_dataset),np.array(y),test_size = .30 , random_state = 0)

print "LogisticRegression Model >>>>>>>>>>>>>>>>>>>>"
logit_clf = LogisticRegression()
logit = ml_model(logit_clf,x_train,x_test,y_train,y_test,x_features)
save_classifier = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\Logistic_Model.pickle","wb")
pickle.dump(logit_clf,save_classifier)
save_classifier.close()

print "SGDClassifier Model >>>>>>>>>>>>>>>>>>>>"
hyperparameters = {"loss":["log"]}
SGDClassifier = SGDClassifier(loss = "log" , penalty = "l2")
SGDClassifier_model = ml_model(SGDClassifier,x_train,x_test,y_train,y_test,x_features)
save_classifier = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\SGDClassifier_model.pickle","wb")
pickle.dump(SGDClassifier,save_classifier)
save_classifier.close()

print "BernoulliNB Model >>>>>>>>>>>>>>>>>>>>"
BernoulliNB = BernoulliNB()
BernoulliNB_model = ml_model(BernoulliNB,x_train,x_test,y_train,y_test,x_features)
save_classifier = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\BernoulliNB_model.pickle","wb")
pickle.dump(BernoulliNB,save_classifier)
save_classifier.close()

print "MultinomialNB Model >>>>>>>>>>>>>>>>>>>>"
MultinomialNB = MultinomialNB()
MultinomialNB_model = ml_model(MultinomialNB,x_train,x_test,y_train,y_test,x_features)
save_classifier = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\MultinomialNB_model.pickle","wb")
pickle.dump(MultinomialNB,save_classifier)
save_classifier.close()


print "RandomForestClassifierModel >>>>>>>>>>>>>>>>>>>>"
RandomForestClassifier = RandomForestClassifier(n_estimators = 200 ,criterion  = "entropy" , max_features  = "sqrt" , max_depth  = 10)
RandomForestClassifier_model = ml_model(RandomForestClassifier,x_train,x_test,y_train,y_test,x_features)
save_classifer = open(r"C:\Users\sauravghosh\Desktop\MachineLearning\Sentimental Analysis\Output\Classifier\RandomForestClassifier_model.pickle","wb")
pickle.dump(RandomForestClassifier,save_classifer)
save_classifier.close()

