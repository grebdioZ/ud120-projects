#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle, sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

def getPOIs():
    result = {}
    for key, value in enron_data.iteritems():
        if value["poi"] == 1:
            result[key] = value
    return result

def getPOINamesFromFile():
    names = []
    with open( "../final_project/poi_names.txt" ) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip("\n")
            if l.startswith( "(y)" ) or l.startswith( "(n)" ):
                names.append( l.split( ") " )[1] )
    return names

pois = getPOIs()

poiNames = getPOINamesFromFile()
#print len(poiNames), poiNames
#print enron_data["PRENTICE JAMES"]
skilling_ceo = enron_data["SKILLING JEFFREY K"]
lay_board = enron_data["LAY KENNETH L"]
fastor_cfo = enron_data["FASTOW ANDREW S"]

numSalary = 0
numEmail = 0
numTotalPayments = 0
for name, person in enron_data.iteritems():
    if person["salary"] != "NaN":
        numSalary += 1
    if person["email_address"] != "NaN":
        numEmail += 1
    if person["total_payments"] != "NaN":
        numTotalPayments += 1

print numSalary, numEmail, numTotalPayments, len(enron_data), 1-float(numTotalPayments)/len(enron_data), len(pois)

