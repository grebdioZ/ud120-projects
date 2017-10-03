#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
data_dict.pop( "TOTAL", 0 )
features = ["salary", "bonus", "bonus"]
data = featureFormat(data_dict, features)


### your code below


for point in data:
    salary = point[0]
    bonus = point[1]
    if bonus > 5000000 and salary> 1000000:
        print point
        boi = bonus
    matplotlib.pyplot.scatter( salary, bonus )

for name, data in data_dict.iteritems():
    if data[ "bonus"] > 5000000 and data[ "salary" ]> 1000000 and data[ "bonus"] != "NaN":
        print name, data[ "bonus"], data[ "salary"]

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
