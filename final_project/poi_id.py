#!/usr/bin/python

import pickle
import sys
from math import sqrt

from final_project.my_helpers import printAllFeaturesAndStatistics, printFeaturesAndStatistics

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages']
# features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi',
#                  'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person',
#                  'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock',
#                  'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers
def removeOutliers(data_dict, features_list):
    return data_dict

data_dict = removeOutliers( data_dict, features_list )

### Task 3: Create new feature(s)
def createNewFeatures(data_dict, features_list):
    #
    def __computeRatio( entry, x, y):
        return float(entry[x])/entry[y]
    #
    def __sqrtOfProduct( entry, x, y):
        return sqrt( float(entry[x])*entry[y] )
    #
    def __identity( entry, x):
        return entry[x]
    #
    def __addCombinedFeature( newFeatureName, computeCombinedFeatureFct, *argFeatNames ):
        for entry in data_dict.values():
            canComputeFeature = True
            for argFeatName in argFeatNames:
                if argFeatName in features_list:
                    features_list.remove( argFeatName )
                if entry.get( argFeatName, "NaN" ) == "NaN":
                    canComputeFeature = False
            if canComputeFeature:
                entry[newFeatureName] = computeCombinedFeatureFct(entry, *argFeatNames)
            else:
                entry[newFeatureName] = "NaN"
        features_list.append( newFeatureName )
    #
    __addCombinedFeature( 'ratio_to_poi', __computeRatio, 'from_this_person_to_poi', 'from_messages' )
    __addCombinedFeature( 'ratio_from_poi', __computeRatio, 'from_poi_to_this_person', 'to_messages' )
    __addCombinedFeature( 'exchange_with_poi', __sqrtOfProduct, 'ratio_to_poi', 'ratio_from_poi')
    return data_dict, features_list

data_dict, features_list = createNewFeatures( data_dict, features_list )
#features_list = ['poi', 'exchange_with_poi', 'shared_receipt_with_poi']

printFeaturesAndStatistics( data_dict, features_list )
#exit()

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
def createClassifier( ):
    def __createGaussianClassifier( *args, **kwargs ):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB( *args, **kwargs )
    def __createDecisionTreeClassifier( *args, **kwargs ):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier( *args, **kwargs )
    def __createSVMClassifier( *args, **kwargs ):
        from sklearn.svm import LinearSVC
        return LinearSVC( *args, **kwargs )
    return __createDecisionTreeClassifier()

# Provided to give you a starting point. Try a variety of classifiers.
clf = createClassifier()

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
def trainClassifier( clf, features, labels ):
    # Example starting point. Try investigating other evaluation techniques!
    from sklearn.cross_validation import train_test_split
    features_train, features_test, labels_train, labels_test = \
        train_test_split(features, labels, test_size=0.3, random_state=42)

trainClassifier( clf, features, labels )

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

import tester
print "Used Features:", features_list
tester.main()