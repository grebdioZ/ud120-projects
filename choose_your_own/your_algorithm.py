#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()

if False:
    ### the training data (features_train, labels_train) have both "fast" and "slow"
    ### points mixed together--separate them so we can give them different colors
    ### in the scatterplot and identify them visually
    grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
    grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
    bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


    #### initial visualization
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
    plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
    plt.legend()
    plt.xlabel("bumpiness")
    plt.ylabel("grade")
    plt.show()
    ################################################################################

### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary
from sklearn import ensemble
from sklearn import tree, svm, naive_bayes
if False:
  clf = tree.DecisionTreeClassifier( max_depth=3, random_state= 0 )
else:
  clf = ensemble.AdaBoostClassifier(
    base_estimator=tree.DecisionTreeClassifier( max_depth=1, min_samples_split=4 ),
    n_estimators=50,
    learning_rate= 1,
    #base_estimator = svm.SVC( ),
    random_state= 0)
clf.fit( features_train, labels_train )
print "Score (train/test): {}/{}".format( clf.score( features_train, labels_train ), clf.score( features_test, labels_test ) )




try:
    prettyPicture(clf, features_train, labels_train)
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass
