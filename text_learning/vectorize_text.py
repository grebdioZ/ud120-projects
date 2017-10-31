#!/usr/bin/python

import os
import pickle
import re
import sys

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText
### in Part 4, do TfIdf vectorization here
from sklearn.feature_extraction.text import TfidfVectorizer

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""

if True:
    print "Reading data"
    word_data = pickle.load( open("your_word_data.pkl") )
    from_data = pickle.load( open("your_email_authors.pkl") )
    v = pickle.load( open("fitted_vectorizer.pkl" ))

else:
    from_sara  = open("from_sara.txt", "r")
    from_chris = open("from_chris.txt", "r")

    from_data = []
    word_data = []

    ### temp_counter is a way to speed up the development--there are
    ### thousands of emails from Sara and Chris, so running over all of them
    ### can take a long time
    ### temp_counter helps you only look at the first 200 emails in the list so you
    ### can iterate your modifications quicker
    temp_counter = 0

    for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
        for path in from_person:
            ### only look at first 200 emails when developing
            ### once everything is working, remove this line to run over full dataset
            temp_counter += 1
            if True or temp_counter < 200:
                #if temp_counter != 153: continue
                path = os.path.join('..', path[:-1])
                #print path
                email = open(path, "r")

                ### use parseOutText to extract the text from the opened email
                text = parseOutText( email )

                ### use str.replace() to remove any instances of the words
                ### ["sara", "shackleton", "chris", "germani"]
                for sigWord in ["sara", "shackleton", "chris", "germani"]:
                  text = text.replace( sigWord, "" )
                  text = text.replace("  ", " ")

                ### append the text to word_data
                word_data.append( text )

                ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
                from_data.append( 0 if from_person=="sara" else 1 )

                email.close()

    print "emails processed"
    from_sara.close()
    from_chris.close()

    pickle.dump( word_data, open("your_word_data.pkl", "w") )
    pickle.dump( from_data, open("your_email_authors.pkl", "w") )

    v = TfidfVectorizer(stop_words="english")
    print "Starting fitting"
    v.fit( word_data )
    #pickle.dump( v, open("fitted_vectorizer.pkl", "w") )

test = v.transform( "give me my money".split(" ") )
print(test.shape)
print(test.toarray())
for y in range(test.shape[0]):
    for x in range(test.shape[1]):
        if test[y,x] > 0:
            print(y, x)
test = v.transform( "give me my money idiot".split(" "))
print(test.shape)
print(test.toarray())
for y in range(test.shape[0]):
    for x in range(test.shape[1]):
        if test[y,x] > 0:
            print(y, x)

print v.get_feature_names()[30100:30110]
print v.idf_[30100:30110]
