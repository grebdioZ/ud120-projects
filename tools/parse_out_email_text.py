#!/usr/bin/python

from nltk.stem.snowball import SnowballStemmer
import string
from nltk.stem.snowball import SnowballStemmer

def parseOutText(f):
    """ given an opened email file f, parse out all text below the
        metadata block at the top
        (in Part 2, you will also add stemming capabilities)
        and return a string that contains all the words
        in the email (space-separated) 
        
        example use case:
        f = open("email_file_name.txt", "r")
        text = parseOutText(f)
        
        """


    f.seek(0)  ### go back to beginning of file (annoying)
    all_text = f.read()

    ### split off metadata
    content = all_text.split("X-FileName:")
    if len(content) > 1:
        stemmedText = performStemming(content[1])

    return stemmedText


def performStemming(text):
    ### remove punctuation
    text_string = text.translate(string.maketrans("", ""), string.punctuation)
    words = [word for word in text_string.split() if word not in ("",)]
    ### split the text string into individual words, stem each word,
    ### and append the stemmed word to words (make sure there's a single
    ### space between each stemmed word)
    words = [SnowballStemmer("english", ignore_stopwords=False).stem(word.strip(" ")) for word in words]
    return " ".join(words)


def main():
    ff = open("../text_learning/test_email.txt", "r")
    text = parseOutText(ff)
    print text



if __name__ == '__main__':
    main()

