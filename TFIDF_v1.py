### ATTEMPT 1: TF-IDF on the comment text

## Rather poor, cos corpus is weak

import numpy as np
import pandas as pd
import re, collections, nltk, math
import sklearn.feature_extraction.text as featExt
from nltk.tokenize import RegexpTokenizer

def isASCIIString(string):
    try:
        string.decode('ascii')
        return True
    except:
        return False


def replaceString(string):
    return string.replace("\\n", " ").replace("&amp;", "&").replace('&#039;', '\'').replace("&quot;", "\"").replace("&lt;", "<").replace("&gt;", ">").strip()
#spellcheck

file = pd.read_csv("comments raw data 220616.txt", delimiter = '~')
messageSet = [replaceString(str(message)) for messageL in file[["message"]].values.tolist() for message in messageL if isASCIIString(replaceString(str(message)))]
tokenizer = RegexpTokenizer(r'\w+')
messageToken = [nltk.word_tokenize(message) for message in messageSet]
NMESSAGES = len(messageToken)

## build counts
wordDict = collections.defaultdict(lambda: 0)
for tokenList in messageToken:
    tokenList = set(tokenList)
    for word in tokenList:
        if not re.match("^[^a-zA-Z0-9]+$", word):
            wordDict[word] += 1

## IDF
idf = {key:math.log(NMESSAGES/(value + 1)) for key, value in wordDict.iteritems()}


## TF-IDF scores, sorted by top 3
finalKeywords = []
for tokenList in messageToken:
    keywords = []
    messageTFIDF = collections.defaultdict(lambda: 0)
    for word in tokenList:
        if not re.match("^[^a-zA-Z0-9]+$", word):
            messageTFIDF[word] += 1
    TFIDFscore = {key:idf[key] * value for key, value in messageTFIDF.iteritems()}
    sortedTFIDF = sorted(TFIDFscore, key = TFIDFscore.get, reverse = True)
    maxIndex = min(3, len(sortedTFIDF))
    for i in range(maxIndex):
        w = sortedTFIDF[i]
        keywords.append((w, TFIDFscore[w]))
    finalKeywords.append(keywords)

### PRINTOUT
print(finalKeywords)
''' 
[[('guess', 6.693323668269949), ('wait', 5.71042701737487), ('mortgage', 5.652489180268651)], 
[('yeeeeee', 7.792761720816526)], 
[('hold', 6.405228458030842), ('asking', 5.846438775057725), ('Was', 5.71042701737487)], 
[('tryin', 7.792761720816526), ('Agency', 7.792761720816526), ('Collect', 7.792761720816526)], 
[('verification', 7.792761720816526), ('trusted', 7.792761720816526), ('linked', 7.387090235656757)], 
[('pursue', 7.387090235656757), ('sue', 7.387090235656757), ('Answer', 7.099201743553092)], 
[('framework', 7.792761720816526), ('decide', 5.998936561946683), ('worth', 5.594711379601839)], 
[('accessories', 7.792761720816526), ('clarifies', 7.792761720816526), ('ultimately', 7.792761720816526)], 
[('participating', 7.792761720816526), ('Want', 6.539585955617669), ('local', 6.0867747269123065)]]
'''
### 