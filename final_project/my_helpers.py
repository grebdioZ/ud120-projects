from numpy import median, std, mean


def printAllFeaturesAndStatistics(data_dict):
    featureValuesWithoutNaN = getFeatureValues(data_dict)
    TOTAL_NUM_VALUES = len( data_dict )
    print("Features: {}".format( featureValuesWithoutNaN.keys() ))
    for feature in featureValuesWithoutNaN.keys():
        values = featureValuesWithoutNaN[ feature ]
        printFeatureInfo(feature, values, TOTAL_NUM_VALUES)


def printFeaturesAndStatistics(data_dict, feature_list):
    featureValuesWithoutNaN = getFeatureValues(data_dict)
    TOTAL_NUM_VALUES = len( data_dict )
    print("Features: {}".format( feature_list ))
    for feature in feature_list:
        values = featureValuesWithoutNaN[ feature ]
        printFeatureInfo(feature, values, TOTAL_NUM_VALUES)

def printFeatureInfo(featureName, values, TOTAL_NUM_VALUES):
    numValues = len(values)
    print("\n*** FEATURE " + featureName + ": ")
    print("Availability: {:.1f}% ({})".format(100.0 * numValues / float(TOTAL_NUM_VALUES), numValues))
    stats = getFeatureStatistics(values)
    if numValues:
        if stats:
            print("Min: {:>13.1f}".format(stats["min"]))
            print("Max: {:>13.1f}".format(stats["max"]))
            print("Avg: {:>13.1f}".format(stats["avg"]))
            print("Med: {:>13.1f}".format(stats["median"]))
            print("Std: {:>13.1f}".format(stats["stddev"]))
        else:
            print("Values: {}".format(values))

def getStatisticsForFeatures(data_dict, feature_list):
    featureValues = getFeatureValues( data_dict )
    result = {}
    for feature in feature_list:
        result[feature] = getFeatureStatistics(featureValues[feature])
    return result

def getFeatureStatistics(values):
    numValues = len(values)
    featureStatistics = dict()
    if numValues:
        if isinstance(values[0], int):
            values = [float(value) for value in values]
        if isinstance(values[0], float):
            featureStatistics["min"] = min(values)
            featureStatistics["max"] = max(values)
            featureStatistics["avg"] = mean(values)
            featureStatistics["median"] = median(values)
            featureStatistics["stddev"] = std(values)
    return featureStatistics

def getFeatureValues(data_dict, allowNaN = False):
    featureValues = {}
    for featName in data_dict.values()[0]:
        featureValues[featName] = []
    for entry_dict in data_dict.values():
        for featName, featValue in entry_dict.iteritems():
            if allowNaN or featValue != "NaN":
                featureValues[featName].append(featValue)
    return featureValues

