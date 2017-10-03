#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list, sort_keys = '../tools/python2_lesson13_keys.pkl')
labels, features = targetFeatureSplit(data)

### your code goes here

from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split( features, labels, test_size = 0.3, random_state=42 )
print "Number of people in test set:", len(features_test)

from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit( features_train, labels_train )
acc = clf.score( features_test, labels_test)

print "Accuracy:", acc

predicted = clf.predict( features_test )
print "Number of POI in Test set:", sum(predicted)
print "Number of True Positives (real POI) in Test set: ", sum([1 if predicted[i] + labels_test[i] == 2 else 0 for i in range(len(labels_test))])

from sklearn.metrics import precision_score, recall_score
print "Precision:", precision_score(labels_test, predicted)
print "Recall:", recall_score(labels_test, predicted)



