#!/usr/bin/python
import copy
import pickle
import sys
from math import sqrt
import random
random.seed(42)
import numpy as np

from final_project.my_helpers import printFeaturesAndStatistics, getStatisticsForFeatures, getFeatureValues

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

g_runInfo = {}
SIMPLE_EVAL = False

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus']
features_list.extend(financial_features)
#features_list.extend(['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees'])
# features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi',
#                  'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person',
#                  'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock',
#                  'director_fees']

def createClassifier( ):
    def __createGaussianClassifier( *args, **kwargs ):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB( priors = (0.99, 0.01), **kwargs )
    def __createDecisionTreeClassifier( *args, **kwargs ):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier( *args, random_state = 42, **kwargs )
    def __createSVMClassifier( *args, **kwargs ):
        from sklearn.svm import SVC
        return SVC( *args, class_weight = "balanced", random_state = 42, **kwargs )
    def __createLinearSVMClassifier( *args, **kwargs ):
        from sklearn.svm import LinearSVC
        return LinearSVC( *args, random_state = 42, **kwargs )
    return __createSVMClassifier( )

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Task 2: Remove outliers

def removeOutliers(data_dict, features_list):
    features_list = copy.copy(features_list)
    features_list.remove( "poi" ) # do not remove outliers in poi...
    g_runInfo["Outlier Removal"] = "avg+-3*stddev for features {}".format( features_list )
    def __isOutlier(value, stats):
        return value != "NaN" and "avg" in stats and \
               (value > stats["avg"] + 3 * stats["stddev"] or
                value < stats["avg"] - 3 * stats["stddev"])
    #
    featureStats = getStatisticsForFeatures( data_dict, features_list )
    for feature in features_list:
        for entry in data_dict.values():
            if __isOutlier( entry[feature], featureStats[feature] ):
                print "Removed outlier of feature {}={:.1f} with stats {}".format( feature, float(entry[feature]), featureStats[feature] )
                entry[ feature ] = "NaN"

#removeOutliers( data_dict, features_list )

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
    def __constZero( entry ):
        return 0
    #
    def __rand( entry ):
        return random.random()
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
g_runInfo["Used Features"] = features_list


def scaleFeatures(data_dict, featuresToScale):
    featuresToScale = copy.copy( featuresToScale )
    if "poi" in featuresToScale:
        featuresToScale.remove( "poi")

    from sklearn.preprocessing import RobustScaler, MinMaxScaler
    values = getFeatureValues( data_dict )
    scaler = RobustScaler(quantile_range=(1, 99))
    #scaler = MinMaxScaler()
    g_runInfo["Feature Scaling"] = "{} on features {}".format(scaler, featuresToScale)
    for feature in featuresToScale:
        scaler.fit( np.array( values[feature] ) )
        for entry in data_dict.values():
            if entry[feature] != "NaN":
                transformed = scaler.transform( [[ entry[feature] ]] )[0][0]
                #print "scaling", feature, entry[feature], transformed
                entry[feature] = transformed

clf = createClassifier()
if clf.__class__.__name__ != "DecisionTreeClassifier":
    scaleFeatures( data_dict, features_list )

#printFeaturesAndStatistics( data_dict, features_list )
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

if SIMPLE_EVAL:
    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

    def validateClassifier(clf, features_test, labels_test):
        from sklearn.cross_validation import train_test_split
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=42)
        clf.fit( features_train, labels_train )
        print "\n----------------------------"
        predicted = clf.predict( features_test )
        print "Actual:    " + "".join([str(int(l)) for l in labels_test])
        print "Predicted: " + "".join([str(int(l)) for l in predicted])
        from sklearn.metrics import precision_score, recall_score, f1_score
        print "* Score:     {:>1.3f}".format( clf.score(features_test, labels_test) )
        print "* Precision: {:>1.3f}".format( precision_score(labels_test, predicted) )
        print "* Recall:    {:>1.3f}".format( recall_score(labels_test, predicted) )
        print "* F1:        {:>1.3f}".format( f1_score(labels_test, predicted) )

    validateClassifier( clf, features_test, labels_test )

else:
    def kFoldValidation( features, labels, N = 3 ):
        from sklearn.cross_validation import KFold
        from sklearn.metrics import precision_score, recall_score, f1_score
        kf = KFold(len(features), N,random_state = 42)
        evalResults = {
            "Accuracy":[],
            "Precision": [],
            "Recall": [],
            "F1": [],
        }
        for train_indices, test_indices in kf:
            features_train = [features[i] for i in train_indices]
            features_test = [features[i] for i in test_indices]
            labels_train = [labels[i] for i in train_indices]
            labels_test = [labels[i] for i in test_indices]
            clf = createClassifier().fit( features_train, labels_train )
            predicted = clf.predict( features_test )
            evalResults["Accuracy"].append( clf.score(features_test, labels_test) )
            evalResults["Precision"].append(precision_score(labels_test, predicted))
            evalResults["Recall"].append(recall_score(labels_test, predicted))
            evalResults["F1"].append(f1_score(labels_test, predicted))

        print "\n----------------------------"
        print("Average scores for {} folds:".format( N ))
        for scoreName in sorted(evalResults.keys()):
            scores = evalResults[scoreName]
            print "* {:<10s}:        {:>1.3f}".format( scoreName, np.mean( scores ) )

    #for n in range(2, 11):
    #    kFoldValidation(features, labels, N=n)
    kFoldValidation(features, labels, N=6)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)

print ""
for name, info in g_runInfo.iteritems():
  print "{}: {}".format(name, info )
print ""
import tester
tester.main()
print "----------------------------\n"
