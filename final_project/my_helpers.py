from numpy import median, std


def printAllFeaturesAndStatistics(data_dict):
    featureValuesWithoutNaN = getValidFeatureValues(data_dict)
    TOTAL_NUM_VALUES = len( data_dict )
    print("Features: {}".format( featureValuesWithoutNaN.keys() ))
    for feature in featureValuesWithoutNaN.keys():
        values = featureValuesWithoutNaN[ feature ]
        printFeatureInfo(feature, values, TOTAL_NUM_VALUES)


def printFeaturesAndStatistics(data_dict, feature_list):
    featureValuesWithoutNaN = getValidFeatureValues(data_dict)
    TOTAL_NUM_VALUES = len( data_dict )
    print("Features: {}".format( feature_list ))
    for feature in feature_list:
        values = featureValuesWithoutNaN[ feature ]
        printFeatureInfo(feature, values, TOTAL_NUM_VALUES)

def printFeatureInfo(featureName, values, TOTAL_NUM_VALUES):
    numValues = len(values)
    print("\n*** FEATURE " + featureName + ": ")
    print("Availability: {:.1f}% ({})".format(100.0 * numValues / float(TOTAL_NUM_VALUES), numValues))
    if numValues:
        if isinstance(values[0], int):
            values = [float(value) for value in values]
        if isinstance(values[0], float):
            print("Min: {:>13.1f}".format(min(values)))
            print("Max: {:>13.1f}".format(max(values)))
            print("Avg: {:>13.1f}".format(sum(values) / float(numValues)))
            print("Med: {:>13.1f}".format(median(values)))
            print("Std: {:>13.1f}".format(std(values)))
        else:
            print("Values: {}".format(values))


def getValidFeatureValues(data_dict):
    featureValuesWithoutNaN = {}
    for featName in data_dict.values()[0]:
        featureValuesWithoutNaN[featName] = []
    for entry_dict in data_dict.values():
        for featName, featValue in entry_dict.iteritems():
            if featValue != "NaN":
                featureValuesWithoutNaN[featName].append(featValue)
    return featureValuesWithoutNaN

