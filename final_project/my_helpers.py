import cPickle
import datetime
import gzip
import json
import os, sys
import time

import numpy
from scipy.stats import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

sys.path.append( "../tools/" )
from parse_out_email_text import performStemming

g_EMAIL_LINK_ROOT = os.path.dirname(__file__) + "/emails_by_address/"
g_EMAIL_ROOT = os.path.dirname(__file__) + "/../"
g_INTERNAL_CATEGORY_TO_UNDERSTANDABLE_KEYWORD = {
    "From": "SENT",
    "To"  : "RCVD",
    "Cc"  : "RCVD-AS-CC",
    "Bcc" : "RCVD-AS-BCC",
}

g_ADDRESS_TAGS = ["From", "To", "Cc", "Bcc"]

from numpy import median, std, mean


def log( *args ):
    print '{:%d-%H:%M.%S:%f}: {}'.format(datetime.datetime.now(), args)

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
    print("/n*** FEATURE " + featureName + ": ")
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


def getNumEmailsFromTo( emails ):
    numMails = 0
    for e in emails:
        for prefix in ("from", "to"):
            fname = g_EMAIL_LINK_ROOT + prefix + "_" + e + ".txt"
            if not os.path.exists( fname ):
                #print("No file found: "+fname)
                pass
            else:
                with open(fname, "r") as f:
                    numMails+=len(f.readlines())
    return numMails

def getEmailFileNames( address, from_or_to ):
    PREFIX_LEN = len("enron_mail_20110402/")
    with open(g_EMAIL_LINK_ROOT + from_or_to + "_" + address + ".txt", "r") as f:
        result = [os.path.abspath(g_EMAIL_ROOT + line[PREFIX_LEN:-2]) for line in f.readlines()]
    return result

def parseEmailListName( emailList):
    addressStartPos = emailList.index("_") + 1
    prefix = emailList[:addressStartPos - 1]
    address = emailList[addressStartPos:emailList.index(".txt")]
    return prefix, address


def parseEmail( fileName ):
    with open( fileName, "r" ) as f:
        s = f.read()
        return parseEmailText(s)



def getBodyStartPosition( emailText ):
    bodyStartPos = emailText.find("X-FileName:")
    if bodyStartPos >= 0:
        bodyStartPos = emailText.find("\n", bodyStartPos)
    return bodyStartPos + 1 if bodyStartPos >= 0 else None


def parseEmailText(s):
    startBodyPos = getBodyStartPosition(s)
    header = s[:startBodyPos]
    info = parseHeader(header)
    info["Body"] = s[startBodyPos:]
    return info


def parseHeader(header):
    def __readInfoFromHeader(paramName, targetParamName = None):
        if targetParamName is None:
            targetParamName = paramName
        lineIndexStart = __readInfoFromHeader.lineIndex
        for line in headerLines[__readInfoFromHeader.lineIndex:]:
            __readInfoFromHeader.lineIndex += 1
            if line.startswith(paramName):
                info[targetParamName] = line[len(paramName) + 2:]
                return
        # not found, reset
        #info[targetParamName] = "" # TODO: Check the best treatment for missing tags here...
        __readInfoFromHeader.lineIndex = lineIndexStart
    #
    info = {}
    __readInfoFromHeader.lineIndex = 0
    header = header.replace("\n\t", "")
    headerLines = header.split("\n")
    __readInfoFromHeader("Message-ID", "ID")
    __readInfoFromHeader("Date")
    __readInfoFromHeader("From")
    __readInfoFromHeader("To")
    __readInfoFromHeader("Subject")
    __readInfoFromHeader("Cc")
    __readInfoFromHeader("Bcc")
    for addressThingy in g_ADDRESS_TAGS:
        cleanUpEmailAddressInfo(info, addressThingy)
    return info


def parseAllEmails( ):
    log("Getting email lists...")
    emailLists = os.listdir(g_EMAIL_LINK_ROOT)
    log("Done. Number of email lists: ", len(emailLists) )
    step = 1
    emailInfo = parseEmailsFromLists(emailLists, step = step)
    saveEmails( emailInfo, "emailInfoRaw.pkl" )
    return emailInfo

def parseEmailsFromTo( emailAddresses):
    listFileNames = []
    for e in emailAddresses:
        for prefix in ("from", "to"):
            fname = prefix + "_" + e + ".txt"
            if os.path.exists(g_EMAIL_LINK_ROOT + fname):
                listFileNames.append(fname)
    log("Number of email lists: ", len(listFileNames) )
    step = 1
    emailInfo = parseEmailsFromLists(listFileNames, step = step)
    return emailInfo

def parseEmailsFromLists(emailListFileNames, step = 1):
    emailInfo = {}
    cnt = 0
    num = numErrors = 0
    numDuplicates = numEmailsStored = 0
    for emailList in emailListFileNames[::step]:
        # for now we only look at the "from" stack to avoid duplicates
        # if not emailList.startswith( "from" ):
        #    continue
        cnt += step
        prefix, address = parseEmailListName(emailList)
        mailFiles = getEmailFileNames(address, prefix)
        # log( "Counter, numEmails=", cnt, len(mailFiles) )
        for fileName in mailFiles[::step]:
            if os.path.isfile(fileName):
                info = parseEmail(fileName)
                cleanUpEmailAddressInfo(info, "From")
                if address.startswith("from") and info["From"] != address:
                    print "Wrong from address? ", address, " != ", info["From"]
                if info["From"] not in emailInfo:
                    emailInfo[info["From"]] = {}
                num += 1
                try:
                    to = info.get("To", "")
                    _ = ("\n*****", cnt, info["ID"], info["From"], to[:min(20, len(to))], info["Date"], info["Subject"],
                         info["Body"][:min(20, len(info["Body"]))])
                except KeyError as e:
                    numErrors += 1
                    print("ERROR: ", fileName, e)
                if emailInfo.get(info["From"], {}).get("ID", None):
                    numDuplicates += 1
                else:
                    numEmailsStored += 1
                    # if "Date" in emailInfo[info["From"]]:
                    emailInfo[info["From"]][info["ID"]] = info
    log("numEmailsStored: {}, numDuplicates eliminated = {}".format(numEmailsStored, numDuplicates))
    log("\n Errors: ", numErrors, "of", num)
    return emailInfo


def cleanUpEmailAddressInfo( info, addressThingy ):
    if addressThingy in info:
        if "<" in info[addressThingy]:
            info[addressThingy] = info[addressThingy].replace("<", "").replace('"', "").replace(' ', "").replace(">", "")
        # separate email addresses by space instead of comma, will simplify correct vectorization
        info[addressThingy] = info[addressThingy].replace(",", " ")


def doStemming(emailInfo):
    log( "Starting doStemming ...")
    cnt = 0
    for address, infos in emailInfo.iteritems():
        for info in infos.values():
            cnt+=1
            for key in info.keys():
                info[key] = info[key].decode('utf-8', 'ignore').encode("utf-8")
            info["Subject"] = performStemming( str( info["Subject"]) )
            info["Body"] = performStemming(str(info["Body"]))
        #if cnt >50:
        #   break

    log("Done doStemming on {} emails...".format(cnt))

def pruneEmails( emailInfo, maxNumEmailsPerAddress ):
    for address, infos in emailInfo.iteritems():
        cnt = 0
        for key in infos.keys():
            cnt += 1
            if cnt > maxNumEmailsPerAddress:
                del infos[key]


def pruneEmailsNew( emailInfoNew, maxNumEmailsPerCategory ):
    for infos in emailInfoNew.values():
        cnt = 0
        for emailID in infos.keys():
            cnt += 1
            if cnt > maxNumEmailsPerCategory:
                del infos[emailID]

def pruneEmailsNewByPerson( emailInfoNew, numPersons ):
    cnt = 0
    for name in emailInfoNew.keys():
        cnt += 1
        if cnt > numPersons:
            del emailInfoNew[name]


def ensureUTF8(emailInfo):
    log( "Starting ensureUTF8...")
    for address, infos in emailInfo.iteritems():
        for info in infos.values():
            for key in info.keys():
                info[key] = info[key].decode('utf-8', 'ignore').encode("utf-8")
    log("Done ensureUTF8 for {} datasets...".format(len(emailInfo)))

def saveEmails( emailInfo, targetFileName ):
    start_s = time.time()
    cPickle.dump(emailInfo, open( targetFileName , "w"))
    log( "Saved {} entries in {} seconds to {}".format( len(emailInfo), time.time() - start_s, targetFileName ) )


def saveAsJSON( data, targetFileName, verbose=False ):
    start_s = time.time()
    json.dump( data, open( targetFileName , "w"), ensure_ascii=True, check_circular=False)
    elapsedTime_s = time.time() - start_s
    if verbose:
        log("Saved data to {} in {} seconds".format( targetFileName, elapsedTime_s ))
    return elapsedTime_s


def loadFromJSON(sourceFileName, verbose = False):
    start_s = time.time()
    data = json.load( open(sourceFileName, "r"))
    elapsedTime_s = time.time() - start_s
    if verbose:
        log("Loaded data from {} in {} seconds".format(sourceFileName, elapsedTime_s))
    return data, elapsedTime_s

def loadFromGzippedJSON(sourceFileName, verbose = False):
    start_s = time.time()
    with gzip.open(sourceFileName, "rb") as f:
        data = json.loads( f.read().decode("ascii"))
    elapsedTime_s = time.time() - start_s
    if verbose:
        log("Loaded data from {} in {} seconds".format(sourceFileName, elapsedTime_s))
    return data, elapsedTime_s


def saveEmailsAsJSON( emailInfo, targetFileName ):
    elapsedTime_s = saveAsJSON( emailInfo, targetFileName )
    log( "Saved {} entries in {} seconds to {}".format( len(emailInfo), elapsedTime_s, targetFileName ) )


def loadEmails( filename ):
    start_s = time.time()
    emailInfo = cPickle.load(open(filename, "r"))
    log("Read {} entries in {} seconds from {}".format( len(emailInfo), time.time() - start_s, filename ) )
    log("Contains {} emails".format( sum([len(info.values()) for info in emailInfo.values()])))
    return emailInfo


def loadEmailsFromJSON(filename):
    emailInfo, elapsedTime_s = loadFromJSON( filename )
    log("Read {} entries in {} seconds from {}".format( len(emailInfo), elapsedTime_s, filename ) )
    numUnique, numTotal = getNumEmails(emailInfo)
    log("Contains {} different emails ({} in total)".format( numUnique, numTotal ) )
    return emailInfo

def getNumEmails(emailInfo):
    ids = set()
    numTotal = 0
    for i1 in emailInfo.values():
        for i2 in i1.values():
            numTotal += len(i2)
            for id in i2.keys():
                ids.add( id )

    numUnique = len(ids)
    return numUnique, numTotal

def getNameForAddress(data_dict, addressToFind):
    for name, values in data_dict.iteritems():
        addr = values.get("email_address", None)
        if addr is not None:
            #print name, addr
            if addr == addressToFind:
                return name
    return None

def reformatEmails(inputFilenameBase):
    """
    Reformats the raw email data created from parsing and stemming:

      [OLD] emailInfos[from_address][emailID] -> dict with parsed email

    into the new format

      [NEW] emailInfoNew[person_name][category][emailID] -> dict with parsed email (To/From/Subject/Body/...)

    where category in ["From", "To", "Cc", "Bcc"] indicates where in the address part the person occurs
    (see also g_INTERNAL_CATEGORY_TO_UNDERSTANDABLE_KEYWORD), and emailID remains the ID parsed from the email.
    This will duplicate some emails (will occur in one person's From category and another ones To category, but
    that does not hurt.
    """
    emailInfo = loadEmailsFromJSON(inputFilenameBase+".json")
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = cPickle.load(data_file)

    newInfos = {}
    cnt = 0
    for infos in emailInfo.values():
        for id, info in infos.iteritems():
            for category in ["From", "To", "Cc", "Bcc"]:
                for emailAddress in info.get(category, "").split(","):
                    name = getNameForAddress(data_dict, emailAddress)
                    if name is not None:
                        cnt+=1
                        if name not in newInfos:
                            newInfos[name] = {}
                        if category not in newInfos[name]:
                            newInfos[name][category] = {}
                        if category not in newInfos[name][category]:
                            newInfos[name][category][id] = {}
                        newInfos[name][category][id] = info
    log("Result should contain", cnt, "emails")
    printInfo( newInfos )
    outputJsonFilename = inputFilenameBase + "Filtered.json"
    saveEmailsAsJSON(newInfos, outputJsonFilename)
    return outputJsonFilename


def printInfo( emailsByPerson ):
    for name, mails in emailsByPerson.iteritems():
        for category in g_ADDRESS_TAGS:
            print name, categoryToUnderstandableFeatureKeyword(category), len(mails.get(category, []))


def computeEmailFeature(emailInfoNew, category, emailProp):
    featureValuesByPerson = __getFeatureValuesByPerson(emailInfoNew, category, emailProp)
    allFeatureValues = featureValuesByPerson.values()
    log("num allFeatureValues:", len(allFeatureValues))
    ### text vectorization--go from strings to lists of numbers
    if emailProp in g_ADDRESS_TAGS:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english',
                                     token_pattern=r"(?u)[\S][\S]+")
    else:
        vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                     stop_words='english' )
    vectorizer.fit( allFeatureValues )
    result = {}
    numFeatures = -1
    for personName, featureData in featureValuesByPerson.iteritems():
        result[personName] = vectorizer.transform( [ featureData ] )
        result[personName] = numpy.array(result[personName].todense().flatten())[0]
        if numFeatures < 0:
            print("numFeatures after vectorization for category {} and prop {}: {}".format( categoryToUnderstandableFeatureKeyword(category), emailProp, len(result[personName])))
            numFeatures = len( result[personName] )
        else:
            assert numFeatures == len( result[personName] )
    return {
        "vectorizedValuesByPerson": result,
        "numFeatures" : numFeatures,
        "vectorizer" : vectorizer
    }


def getEmailFeatureNamePrefix(category, feature):
    return "emails_{}_{}".format( categoryToUnderstandableFeatureKeyword( category ), feature )


def categoryToUnderstandableFeatureKeyword( category ):
    return g_INTERNAL_CATEGORY_TO_UNDERSTANDABLE_KEYWORD[ category ]


def understandableFeatureKeywordToCategory( categoryKeyword ):
    for category, understandable in g_INTERNAL_CATEGORY_TO_UNDERSTANDABLE_KEYWORD.iteritems():
        if categoryKeyword == understandable:
            return category
    raise RuntimeError("Non-existing categoryKeyword: {}".format( categoryKeyword ))


def convertEmailFeatureNames(newFeatureValues):
    """
    Converts old email cachefile (with confusing from/to categories and thus feature names into new ones with SENT/RCVD
    """
    def __replOldWithNewFName(f):
        for oldKey in g_INTERNAL_CATEGORY_TO_UNDERSTANDABLE_KEYWORD.keys():
            # print "emails_{}_".format( oldKey ), "emails_{}_".format( categoryToUnderstandableFeatureKeyword( oldKey ) )
            searchString = "emails_{}_".format(oldKey)
            if searchString in f:
                return f.replace(searchString, "emails_{}_".format(categoryToUnderstandableFeatureKeyword(oldKey)))
        return f

    oldEmailFeatureCacheFileName = "EmailFeatures_To-From_From-To_To-Subject_From-Subject.json"
    log("Loading stored email features from " + oldEmailFeatureCacheFileName)
    (newFeatureValues, _), _ = loadFromJSON(oldEmailFeatureCacheFileName, verbose=True)
    for featuresByPerson in newFeatureValues.values():
        keys = list(featuresByPerson.keys())
        for key in keys:
            # print("replacing {} by {}".format( key, __replOldWithNewFName(key) ))
            featuresByPerson[__replOldWithNewFName(key)] = featuresByPerson.pop(key)

    saveAsJSON(newFeatureValues, "EmailFeatures_TO_FROM_SUBJECTS.json", verbose=True)


def selectFeatures(features_train, labels_train, features_test, percentile=10, runInfo=None):
    selector = SelectPercentile(f_classif, percentile=percentile)
    selector.fit(features_train, labels_train)
    features_train_transformed = selector.transform(features_train)
    features_test_transformed = selector.transform(features_test)
    if runInfo is not None:
        runInfo["Selected Features:"] = "Perc = {}, Num = {}".format(percentile, len(features_train_transformed[0]))
    return features_train_transformed, features_test_transformed


def summarizeFeatureList( features_list ):
    sumFeat = set()
    emailFeats = {}
    for f in features_list:
        if f == 'poi':
            continue
        if f.startswith("emails_"):
            fStem = f[:f.find("_", f.find("_", f.find("_")+1)+1)]
            if fStem in emailFeats:
                emailFeats[fStem] += 1
            else:
                emailFeats[fStem] = 1
        else:
            sumFeat.add( f )

    for name, frequency in emailFeats.iteritems():
        sumFeat.add( "{} (x{})".format( name, frequency ) )
    return sorted(list(sumFeat))


def getListFromDictOfLists(listOfLists):
    accuList = []
    for elemList in listOfLists.values():
        accuList.extend(elemList)
    return accuList


def __getFeatureValuesByPerson(emailInfoNew, category, emailProp):
    featureValuesByPerson = {}
    for personName, emailInfo in emailInfoNew.iteritems():
        featureValuesByPerson[personName] = ""
        if category in emailInfo:
            for email in emailInfo[category].values():
                if emailProp in email.keys():
                    val = email[emailProp]
                    if emailProp == "Subject" and "org" in val: # ignore "org", seems pointless
                        continue
                    featureValuesByPerson[personName] += " " + val
    return featureValuesByPerson


def computeExternalTestResult( clf, data, feature_list ):
    import tester
    try:
        return tester.main( clf=clf, dataset=data, feature_list=feature_list)
    except ValueError as e:
        print("ERROR: Exception occurred running tester: {}".format( e ))
        return { "Accuracy":0, "F1":0, "Precision":0, "Recall":0 }


def shortenString( string, maxLen=200, giveTruncationInfo=True ):
    result = string[:maxLen]
    if len(string) > maxLen:
        result += "..."
        if giveTruncationInfo:
            result += " ({} characters truncated)".format( len(string) - maxLen )
    return result


def removeFeatures(featList, featuresToRemove):
    for f in featuresToRemove:
        featList.remove(f)
    print("New feature list after removeFeatures: " + str(featList[1:]))

