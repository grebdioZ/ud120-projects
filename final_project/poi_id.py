#!/usr/bin/python
import copy
import pickle
import sys
from math import sqrt
import random
random.seed(42)
import numpy as np

from final_project.my_helpers import *

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

g_runInfo = {}
g_computationTimeInfo = [
    ("Start", time.time())
]
g_RUN_PARAMS = {}

SIMPLE_EVAL = False
RECOMPUTE_EMAILS = False
DUMP_RESULTS = False
COMPUTE_PRUNED_EMAIL_VERSION_FOR_FAST_TESTING = False
USE_PRUNED_EMAIL_VERSION_FOR_FAST_TESTING = False
OVERWRITE_INITIAL_FEATURE_LIST_FOR_TESTING = False

LOAD_EMAIL_FEATURES = True
SAVE_EMAIL_FEATURES = False
RUN_EXTERNAL_VALIDATION = True
g_RUN_PARAMS["MIN_FEATURE_IMPORTANCE"] = 1e-6
g_RUN_PARAMS["NUM_VALIDATION_FOLDS"] = 6
g_RUN_PARAMS["NUM_UNIMPORTANT_FEATURES_TO_TRY_REMOVAL"] = 5
g_RUN_PARAMS["MAX_ALLOWED_OPT_CRIT_DECREASE"] = 0.2
g_RUN_PARAMS["OPTIMIZATION_CRIT"] = "F1_ext"

EMAIL_DETAIL_FEATURES = [
    #(understandableFeatureKeywordToCategory( "RCVD" ), "From" ),
    #(understandableFeatureKeywordToCategory( "SENT" ), "To" ),
    #(understandableFeatureKeywordToCategory( "RCVD" ), "Subject" ),
    (understandableFeatureKeywordToCategory( "SENT" ), "Subject" ),
]


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi']
email_features = ['shared_receipt_with_poi', 'from_poi_to_this_person', 'from_this_person_to_poi', 'to_messages', 'from_messages']
#email_features = [ 'to_messages', 'from_messages' ]
features_list.extend(email_features)
financial_features = []
#financial_features = ["deferred_income", "expenses"]
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees']
#financial_features = ['bonus', 'expenses', 'total_stock_value', 'restricted_stock']#, 'salary', 'total_payments']
features_list.extend(financial_features)
#features_list.extend(['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'total_stock_value', 'long_term_incentive', 'exercised_stock_options', 'deferred_income', 'expenses', 'restricted_stock', 'director_fees'])
# features_list = ['poi', 'salary', 'to_messages', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus',
#                  'restricted_stock_deferred', 'total_stock_value', 'shared_receipt_with_poi',
#                  'long_term_incentive', 'exercised_stock_options', 'from_messages', 'other', 'from_poi_to_this_person',
#                  'from_this_person_to_poi', 'deferred_income', 'expenses', 'restricted_stock',
#                  'director_fees']


def addComputationTimeInfo( text ):
    global g_computationTimeInfo
    g_computationTimeInfo.append( (text, time.time()) )


def printOverallTimeInfo():
    total_s = int( round( g_computationTimeInfo[-1][1] - g_computationTimeInfo[0][1] ) )
    print( "OVERALL COMPUTATION TIME: {} minutes and {} seconds".format( int( total_s ) // 60, int(total_s) % 60 ) )


def printDetailedComputationTimeInfo():
    prevTime_s = 0
    print ("*** TIME INFO START *** ")
    for msg, time_s in g_computationTimeInfo:
        if prevTime_s > 0:
            timeDiff_s = time_s - prevTime_s
            print( "STEP {:>30} took {} minutes and {} seconds".format( msg, int( timeDiff_s ) // 60, int(timeDiff_s) % 60 ) )
        prevTime_s = time_s
    printOverallTimeInfo()
    print ("*** TIME INFO END *** ")


def createClassifier( ):
    def __createGaussianClassifier( ):
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB( priors = (0.97, 0.03) )
    def __createDecisionTreeClassifier( ):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier( random_state = 42, min_samples_leaf=4, criterion='entropy' )
    def __createSVMClassifier( ):
        from sklearn.svm import SVC
        return SVC( class_weight = "balanced", random_state = 42 )
    def __createLinearSVMClassifier( ):
        from sklearn.svm import LinearSVC
        return LinearSVC( random_state = 42)
    return __createDecisionTreeClassifier( )

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

addComputationTimeInfo( "Initial Data Load" )

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
#addComputationTimeInfo("Outlier Removal")

def addEmailFeatures( data_dict, features_list ):
    def __computeEmailInfo():
        emails = [i["email_address"] for i in data_dict.values() if i["email_address"] != "NaN"]
        emailInfo = parseEmailsFromTo(emails)
        ensureUTF8(emailInfo)
        saveEmailsAsJSON(emailInfo, "emailInfoSubjectsRaw.json")
        # loadEmailsFromJSON( "emailInfoSubjectsRaw.json" )
        doStemming(emailInfo)
        saveEmailsAsJSON(emailInfo, "emailInfoSubjectsStemmed.json")
        reformatEmails("emailInfoSubjectsStemmed")

    def __addFeature( category, featureName ):
        log("Creating email feature for category {}: {}".format( categoryToUnderstandableFeatureKeyword( category ), featureName) )
        newEmailFeature = computeEmailFeature( emailInfoNew, category, featureName )
        for i in range(newEmailFeature["numFeatures"]):
            featureID = newEmailFeature["vectorizer"].get_feature_names()[i]
            featureNameInDataDict = "{}_{}".format( getEmailFeatureNamePrefix( category, featureName ), featureID )
            for name in data_dict.keys():
                if name not in newFeatureValues:
                    newFeatureValues[ name ] = {}
                try:
                    newFeatureValues[name][featureNameInDataDict] = newEmailFeature["vectorizedValuesByPerson"][name][i]
                except:
                    newFeatureValues[name][featureNameInDataDict] = 0.0
                #if entry[featureNameInDataDict] > 0.0:
                #    print name, featureNameInDataDict, entry[featureNameInDataDict]

    #emailFeatureCacheFileName = "EmailFeatures_To-From_From-To_To-Subject_From-Subject.json" #"EmailFeatures_{}.json".format( "_".join( ["{}-{}".format( category, feature) for category, feature in EMAIL_DETAIL_FEATURES] ) )
    emailFeatureCacheFileName = "EmailFeatures_TO_FROM_SUBJECTS.json"
    if LOAD_EMAIL_FEATURES and os.path.isfile( emailFeatureCacheFileName ):
        log("Loading stored email features from " + emailFeatureCacheFileName )
        newFeatureValues, _ = loadFromJSON( emailFeatureCacheFileName, verbose=True)
    else:
        log("Computing email features...")
        if RECOMPUTE_EMAILS:
            __computeEmailInfo()

        if COMPUTE_PRUNED_EMAIL_VERSION_FOR_FAST_TESTING:
            emailInfoNew = loadEmailsFromJSON("emailInfoSubjectsStemmedFiltered.json")
            pruneEmailsNewByPerson( emailInfoNew, 10 )
            saveEmailsAsJSON(emailInfoNew, "emailInfoSubjectsStemmedFilteredPruned2.json")
            emailInfoNew = loadEmailsFromJSON("emailInfoSubjectsStemmedFilteredPruned2.json")

        if USE_PRUNED_EMAIL_VERSION_FOR_FAST_TESTING:
            g_runInfo["EmailDataWarning"] = "WARNING: Have only used subset of emails to speed up processing! Disable USE_PRUNED_EMAIL_VERSION_FOR_FAST_TESTING for full eval."
            emailInfoNew = loadEmailsFromJSON("emailInfoSubjectsStemmedFilteredPruned2.json")
        else:
            emailInfoNew = loadEmailsFromJSON("emailInfoSubjectsStemmedFiltered.json")
        newFeatureValues = {}
        for category, featuresByPerson in EMAIL_DETAIL_FEATURES:
            __addFeature( category, featuresByPerson )
        if SAVE_EMAIL_FEATURES:
            saveAsJSON( newFeatureValues, emailFeatureCacheFileName )

    # Merge with existing features
    features_list.extend( getFeatureListForEmailFeatures(newFeatureValues) )
    for name, data in data_dict.iteritems():
        data.update( newFeatureValues[ name ] )
    addComputationTimeInfo("Create email features")



def getFeatureListForEmailFeatures( dataDict ):
    fList = []
    featPrefixes = []
    for category, feature in EMAIL_DETAIL_FEATURES:
        featPrefixes.append( getEmailFeatureNamePrefix( category, feature ) )
    featureNames = []
    for name, data in dataDict.iteritems():
        featureNames = data.keys()
        break
    for featureName in featureNames:
        for pref in featPrefixes:
            if featureName.startswith( pref ):
                fList.append(featureName)
                break
    return fList


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
    def __addCombinedFeatures( ):
        __addCombinedFeature( 'ratio_to_poi', __computeRatio, 'from_this_person_to_poi', 'to_messages' )
        __addCombinedFeature( 'ratio_from_poi', __computeRatio, 'from_poi_to_this_person', 'from_messages' )
        __addCombinedFeature( 'exchange_with_poi', __sqrtOfProduct, 'ratio_to_poi', 'ratio_from_poi')
        addComputationTimeInfo("Create combined features")

    #__addCombinedFeatures()
    if EMAIL_DETAIL_FEATURES:
        addEmailFeatures(data_dict, features_list)
    return data_dict, features_list

data_dict, features_list = createNewFeatures( data_dict, features_list )

#features_list = ['poi', 'exchange_with_poi', 'shared_receipt_with_poi']


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

    addComputationTimeInfo("Scaled Features")

if createClassifier().__class__.__name__ != "DecisionTreeClassifier":
    scaleFeatures( data_dict, features_list )

#printFeaturesAndStatistics( data_dict, features_list )
#exit()

### Store to my_dataset for easy export below.
my_dataset = data_dict

# overwrite feature list for testing purposes:
if OVERWRITE_INITIAL_FEATURE_LIST_FOR_TESTING:
    features_list = [
        "poi",
        "exchange_with_poi",
        "shared_receipt_with_poi",
        "emails_From_Subject_confidenti",
        "emails_To_Subject_product",
        "emails_From_To_sally.beck@enron.com",
        "emails_To_From_vince.kaminski@enron.com",
        "emails_To_From_john.lavorato@enron.com",
        "emails_To_From_mike.mcconnell@enron.com",
    ]

initialFeatureListInfo = features_list[1:] if len( features_list ) <= 30 else summarizeFeatureList(features_list)
g_runInfo["FEATURES INITIALLY USED for evaluation"] = "Num = {}: {}".format(len(features_list) - 1, initialFeatureListInfo )


def ClfResults(featureList):
    return {
        "Accuracy": 0,
        "F1": 0,
        "Precision": 0,
        "Recall": 0,
        "F1_ext": 0,
        "features_list": copy.copy(featureList),
        "features_importances": [(f, -1) for f in featureList],
    }


bestResult = ClfResults(features_list)


def trainAndEvaluate(featureList, bestResult, allowRecursion=True):
    ### Extract features and labels from dataset for local testing
    if not allowRecursion:
        print("")
        print("******* trainAndEvaluate: probing only w/o recursion ************** on features {}".format( featureList[1:] ))
    else:
        print("")
        print("******************************************")
        print("* trainAndEvaluate (current best: {:.3f}) *".format( bestResult[ g_RUN_PARAMS["OPTIMIZATION_CRIT"]] ))
        print("******************************************")

    data = featureFormat(my_dataset, featureList, sort_keys=True)
    labels, features = targetFeatureSplit(data)
    addComputationTimeInfo("Target Feature Split")


    ### Task 4: Try a varity of classifiers
    ### Please name your classifier clf for easy export below.
    ### Note that if you want to do PCA or other multi-stage operations,
    ### you'll need to use Pipelines. For more info:
    ### http://scikit-learn.org/stable/modules/pipeline.html

    # if SIMPLE_EVAL:
    #
    #     def validateClassifier(features, labels):
    #         from sklearn.cross_validation import train_test_split
    #         features_train, features_test, labels_train, labels_test = \
    #             train_test_split(features, labels, test_size=0.3, random_state=42)
    #         addComputationTimeInfo("Train/Test Split")
    #         clf = createClassifier().fit( features_train, labels_train )
    #         addComputationTimeInfo("Simple Fit")
    #         print "\n----------------------------"
    #         predicted = clf.predict( features_test )
    #         print "Actual:    " + "".join([str(int(l)) for l in labels_test])
    #         print "Predicted: " + "".join([str(int(l)) for l in predicted])
    #         from sklearn.metrics import precision_score, recall_score, f1_score
    #         scores = {
    #             "Accuracy": clf.score(features_test, labels_test),
    #             "Precision": precision_score(labels_test, predicted),
    #             "Recall": recall_score(labels_test, predicted),
    #             "F1": f1_score(labels_test, predicted),
    #         }
    #         for name, value in scores.iteritems():
    #             print "* {:<10s}: {:>1.3f}".format( name, value )
    #         addComputationTimeInfo("Simple Eval")
    #         return clf, scores
    #
    #     clf, scores = validateClassifier( features, labels )
    #     bestResult.update( scores )

    def kFoldValidation( features, labels, N=3 ):
        def __updateEvalResults(trainIndices, testIndices):
            features_train = [features[i] for i in trainIndices]
            features_test = [features[i] for i in testIndices]
            labels_train = [labels[i] for i in trainIndices]
            labels_test = [labels[i] for i in testIndices]
            #features_train, features_test = selectFeatures(features_train, labels_train, features_test, percentile=20, runInfo=g_runInfo)
            classifier.fit( features_train, labels_train )
            evalResults["Importances"].append( classifier.feature_importances_ )
            predicted = classifier.predict( features_test )
            evalResults["Accuracy"].append( classifier.score(features_test, labels_test) )
            evalResults["Precision"].append(precision_score(labels_test, predicted))
            evalResults["Recall"].append(recall_score(labels_test, predicted))
            f1 = f1_score(labels_test, predicted)
            evalResults["F1"].append(f1)

        def __computeMeanScores(meanScores, evalResults):
            for scoreName in sorted(evalResults.keys()):
                if scoreName not in ("Importances",):
                    meanScores[scoreName] = np.mean(evalResults[scoreName])
                    if allowRecursion:
                        print "* {:<10s}: {:>1.3f}".format(scoreName, meanScores[scoreName])
            # Compute average importances
            meanScores["Importances"] = []
            for featureIndex in range(len(featureList) - 1):
                featVec = []
                for impVec in evalResults["Importances"]:
                    featVec.append(impVec[featureIndex])
                meanScores["Importances"].append(np.mean(featVec))

        from sklearn.cross_validation import KFold
        from sklearn.metrics import precision_score, recall_score, f1_score
        kf = KFold(len(features), N,random_state = 42)
        evalResults = {
            "Accuracy":[],
            "Precision": [],
            "Recall": [],
            "F1": [],
            "Importances": []
        }
        classifier = None
        for train_indices, test_indices in kf:
            classifier = createClassifier()
            __updateEvalResults(train_indices, test_indices)
        if allowRecursion:
            addComputationTimeInfo( "kFold Validation" )
            print "\n----------------------------"
            print("Average scores for {} folds:".format( N ))
        meanScores = {}
        __computeMeanScores(meanScores, evalResults)

        featuresByAverageImportance = sorted(
            [(featureList[index + 1], imp) for index, imp in enumerate(meanScores["Importances"])],
            key=lambda x: -x[1])
        return classifier, meanScores, featuresByAverageImportance

    clf, scores, featuresByImportance = kFoldValidation(features, labels, N=g_RUN_PARAMS["NUM_VALIDATION_FOLDS"])

    ### Task 5: Tune your classifier to achieve better than .3 precision and recall
    ### using our testing script. Check the tester.py script in the final project
    ### folder for details on the evaluation method, especially the test_classifier
    ### function. Because of the small size of the dataset, the script uses
    ### stratified shuffle split cross validation. For more info:
    ### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
    if clf.__class__.__name__ != "DecisionTreeClassifier":
        # no need to iterate
        print("No need to iterate")
        return bestResult
    else:
        #featuresByImportance = sorted(
        #    [(features_list[index + 1], imp) for index, imp in enumerate(clf.feature_importances_ )],
        #    key=lambda x: -x[1])
        numFeatures = len(featureList) - 1 # minus poi
        featWithNonZeroImportances = [(feat, round(imp, 3)) for feat, imp in featuresByImportance if imp > 0]
        print ("Importances >0: ", featWithNonZeroImportances)
        if g_RUN_PARAMS["OPTIMIZATION_CRIT"] == "F1_ext":
            if len( featWithNonZeroImportances ):
                # will raise error otherwise
                scores["F1_ext"] = computeExternalTestResult( clf, data_dict, featureList )
            else:
                scores["F1_ext"] = 0.0
        scores["clf"] = clf
        scores["features_list"] = copy.copy(featureList)
        scores["features_importances"] = copy.copy(featuresByImportance)
        newResultsTooBad = shouldStop( scores, bestResult )
        updateBestResult( bestResult, scores )
        if not allowRecursion:
            print("No recursion allowed, returning.")
            return bestResult
        if newResultsTooBad:
            print("Results have become much worse than with previous feature set, aborting recursion!")
            return bestResult
        print( "Trying to remove all features with imp < {}".format( g_RUN_PARAMS["MIN_FEATURE_IMPORTANCE"] ) )
        featuresToRemove = [feature for feature, imp in featuresByImportance if imp < g_RUN_PARAMS["MIN_FEATURE_IMPORTANCE"] ]
        if featuresToRemove and len( featuresToRemove ) < numFeatures:
            removeFeatures(featureList, featuresToRemove)
            return trainAndEvaluate(featureList, bestResult)
        else:
            if numFeatures > 1:
                print("Threshold removed none or all features, trying removal of each of the {} least important ones.".format( g_RUN_PARAMS["NUM_UNIMPORTANT_FEATURES_TO_TRY_REMOVAL"] ))
                bestBranchResult = ClfResults(featureList)
                numBranches = min(numFeatures - 1, g_RUN_PARAMS["NUM_UNIMPORTANT_FEATURES_TO_TRY_REMOVAL"])
                for i in range( numBranches ):
                    print( "No feature with with imp below threshold found, removing the one with index {}: {}".format(-i, featuresByImportance[-i]
                         ))
                    branchFeatureList = copy.copy(featureList)
                    removeFeatures(branchFeatureList, [featuresByImportance[-i][0]])
                    bestBranchResult = trainAndEvaluate(branchFeatureList, bestBranchResult, allowRecursion=False )
                print("Best result had {} of {} for features {}".format( g_RUN_PARAMS["OPTIMIZATION_CRIT"], bestBranchResult[g_RUN_PARAMS["OPTIMIZATION_CRIT"]], bestBranchResult["features_list"]))
                if not shouldStop( bestBranchResult, bestResult ):
                    return trainAndEvaluate(bestBranchResult["features_list"], bestResult, allowRecursion=True)
                else:
                    print("No sub-branch warranted further investigation, stopping.")
            else:
                print("No more features to remove, stopping.")
            print("Best {} remains {} with features {}".format( g_RUN_PARAMS["OPTIMIZATION_CRIT"], bestResult[g_RUN_PARAMS["OPTIMIZATION_CRIT"]], bestResult["features_list"][1:] ) )
            return bestResult


def shouldStop( newResults, bestResults ):
    return (bestResults[g_RUN_PARAMS["OPTIMIZATION_CRIT"]] - newResults[g_RUN_PARAMS["OPTIMIZATION_CRIT"] ]
            > g_RUN_PARAMS[ "MAX_ALLOWED_OPT_CRIT_DECREASE"] )


def updateBestResult(bestResult, scores ):
    optParam = g_RUN_PARAMS["OPTIMIZATION_CRIT"]
    if scores[optParam] > bestResult[optParam]:
        print(
        "Best {} improved from {} to {} with features {}".format(optParam,
                                                                 round(bestResult[optParam], 3), round(scores[optParam], 3),
                                                                 scores["features_list"][1:]))
        bestResult.update(scores)
        assert len(bestResult["features_list"]) == len(bestResult["features_importances"]) + 1
    else:
        print("Best {} of {} could not be improved (New: {} with features {})".format(optParam,
                                                                                      round(bestResult[optParam], 3),
                                                                                      round(scores[optParam], 3),
                                                                                      scores["features_list"][1:]))


bestResult = trainAndEvaluate( features_list, bestResult )

clf = bestResult["clf"]
features_list = bestResult["features_list"]
g_runInfo["FEATURES IN BEST CLASSIFIER, with importances"] = "Num = {}: {}".format(
    len( bestResult["features_importances"] ), [ ( name, round(imp, 3) ) for name, imp in bestResult["features_importances"] ] )

print "\n----------------------------"
print("Best result scores ( averages for {} folds, unless SIMPLE_EVAL was used ):".format(g_RUN_PARAMS["NUM_VALIDATION_FOLDS"]))
meanScores = {}
for scoreName in ["Accuracy", "F1", "Precision", "Recall"]:
    print "* {:<10s}: {:>1.3f}".format(scoreName, bestResult[scoreName])

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print("")
for name, info in sorted( g_RUN_PARAMS.iteritems() ):
    print "{}: {}".format(name, info)

for name, info in sorted( g_runInfo.iteritems() ):
    print "\n{}: {}".format(name, info)
print("")

if DUMP_RESULTS:
    dump_classifier_and_data(clf, my_dataset, features_list)
    addComputationTimeInfo("dump_classifier_and_data")

if RUN_EXTERNAL_VALIDATION:
    computeExternalTestResult( clf, my_dataset, features_list )
    addComputationTimeInfo("External Validation (tester)")
printOverallTimeInfo()
print "----------------------------\n"

#printDetailedComputationTimeInfo()
