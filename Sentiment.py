#!/usr/bin/env python
import re, random, math, collections, itertools

PRINT_ERRORS=1

#------------- Function Definitions ---------------------


def readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia):

    #reading pre-labeled input and splitting into lines
    posSentences = open('rt-polarity.pos', 'r', encoding="ISO-8859-1")
    posSentences = re.split(r'\n', posSentences.read())

    negSentences = open('rt-polarity.neg', 'r', encoding="ISO-8859-1")
    negSentences = re.split(r'\n', negSentences.read())

    posSentencesNokia = open('nokia-pos.txt', 'r')
    posSentencesNokia = re.split(r'\n', posSentencesNokia.read())

    negSentencesNokia = open('nokia-neg.txt', 'r', encoding="ISO-8859-1")
    negSentencesNokia = re.split(r'\n', negSentencesNokia.read())
 
    posDictionary = open('positive-words.txt', 'r', encoding="ISO-8859-1")
    posWordList = []
    for line in posDictionary:
        line = line.strip().lower()
        if line and not line.startswith(";"):
            posWordList.append(line)

    negDictionary = open('negative-words.txt', 'r', encoding="ISO-8859-1")
    negWordList = []
    for line in negDictionary:
        line = line.strip().lower()
        if line and not line.startswith(";"):
            negWordList.append(line)

    for i in posWordList:
        sentimentDictionary[i] = 1
    for i in negWordList:
        sentimentDictionary[i] = -1

    # Create Training and Test Datsets:
    # We want to test on sentences we haven't trained on, 
    # to see how well the model generalses to previously unseen sentences

    # create 90-10 split of training and test data from movie reviews, with sentiment labels    
    for i in posSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="positive"
        else:
            sentencesTrain[i]="positive"

    for i in negSentences:
        if random.randint(1,10)<2:
            sentencesTest[i]="negative"
        else:
            sentencesTrain[i]="negative"

    # create Nokia Datset:
    for i in posSentencesNokia:
            sentencesNokia[i]="positive"
    for i in negSentencesNokia:
            sentencesNokia[i]="negative"

#----------------------------End of data initialisation ----------------#

# calculates p(W|Positive), p(W|Negative) and p(W) for all words in training data
def trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord):
    posFeatures = [] # [] initialises a list [array]
    negFeatures = [] 
    freqPositive = {} # {} initialises a dictionary [hash function]
    freqNegative = {}
    dictionary = {}
    posWordsTot = 0
    negWordsTot = 0
    allWordsTot = 0

    # iterate through each sentence/sentiment pair in the training data
    for sentence, sentiment in sentencesTrain.items():
        wordList = re.findall(r"[\w']+", sentence)
        
        for word in wordList: # calculate over unigrams
            allWordsTot += 1 # keeps count of total words in dataset
            if not (word in dictionary):
                dictionary[word] = 1
            if sentiment=="positive" :
                posWordsTot += 1 # keeps count of total words in positive class

                # keep count of each word in positive context
                if not (word in freqPositive):
                    freqPositive[word] = 1
                else:
                    freqPositive[word] += 1    
            else:
                negWordsTot+=1 # keeps count of total words in negative class
                
                # keep count of each word in positive context
                if not (word in freqNegative):
                    freqNegative[word] = 1
                else:
                    freqNegative[word] += 1

    for word in dictionary:
        # do some smoothing so that minimum count of a word is 1
        if not (word in freqNegative):
            freqNegative[word] = 1
        if not (word in freqPositive):
            freqPositive[word] = 1

        # Calculate p(word|positive)
        pWordPos[word] = freqPositive[word] / float(posWordsTot)

        # Calculate p(word|negative) 
        pWordNeg[word] = freqNegative[word] / float(negWordsTot)

        # Calculate p(word)
        pWord[word] = (freqPositive[word] + freqNegative[word]) / float(allWordsTot) 

#---------------------------End Training ----------------------------------

# implement naive bayes algorithm
# INPUTS:
#   sentencesTest is a dictonary with sentences associated with sentiment 
#   dataName is a string (used only for printing output)
#   pWordPos is dictionary storing p(word|positive) for each word
#      i.e., pWordPos["apple"] will return a real value for p("apple"|positive)
#   pWordNeg is dictionary storing p(word|negative) for each word
#   pWord is dictionary storing p(word)
#   pPos is a real number containing the fraction of positive reviews in the dataset
def testBayes(sentencesTest, dataName, pWordPos, pWordNeg, pWord,pPos):

    print("Naive Bayes classification")
    pNeg=1-pPos

    # These variables will store results
    total=0
    correct=0
    totalpos=0
    totalpospred=0
    totalneg=0
    totalnegpred=0
    correctpos=0
    correctneg=0

    # for each sentence, sentiment pair in the dataset
    for sentence, sentiment in sentencesTest.items():
        wordList = re.findall(r"[\w']+", sentence)#collect all words

        pPosW=pPos
        pNegW=pNeg

        for word in wordList: # calculate over unigrams
            if word in pWord:
                if pWord[word]>0.00000001:
                    pPosW *=pWordPos[word]
                    pNegW *=pWordNeg[word]

        prob=0;            
        if pPosW+pNegW >0:
            prob=pPosW/float(pPosW+pNegW)


        total+=1
        if sentiment=="positive":
            totalpos+=1
            if prob>0.5:
                correct+=1
                correctpos+=1
                totalpospred+=1
            else:
                correct+=0
                totalnegpred+=1
                if PRINT_ERRORS:
                    print ("ERROR (pos classed as neg %0.2f):" %prob + sentence)
        else:
            totalneg+=1
            if prob<=0.5:
                correct+=1
                correctneg+=1
                totalnegpred+=1
            else:
                correct+=0
                totalpospred+=1
                if PRINT_ERRORS:
                    print ("ERROR (neg classed as pos %0.2f):" %prob + sentence)
 
 
# TODO for Step 2: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
 
    # accuracy
    accuracy = correct / float(total)

    # precision and recall for the positive class
    precision_positive = correctpos / float(totalpospred)
    recall_positive = correctpos / float(totalpos)

    # precision and recall for the negative class
    precision_negative = correctneg / float(totalnegpred)
    recall_negative = correctneg / float(totalneg)

    # F1 scores
    f1_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
    f1_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)
    f1_average = (f1_positive + f1_negative) / 2.0

    print("\nEvaluation Metrics:")
    print("Accuracy:", round(accuracy, 3))
    print("Precision (Positive):", round(precision_positive, 3), "Recall (Positive):", round(recall_positive, 3))
    print("Precision (Negative):", round(precision_negative, 3), "Recall (Negative):", round(recall_negative, 3))
    print("F1 Score (Averaged):", round(f1_average, 3))




# This is a simple classifier that uses a sentiment dictionary to classify 
# a sentence. For each word in the sentence, if the word is in the positive 
# dictionary, it adds 1, if it is in the negative dictionary, it subtracts 1. 
# If the final score is above a threshold, it classifies as "Positive", 
# otherwise as "Negative"
def testDictionary(sentencesTest, dataName, sentimentDictionary, threshold):

    print("Dictionary-based classification")
    total=0
    correct=0
    totalpos=0
    totalneg=0
    totalpospred=0
    totalnegpred=0
    correctpos=0
    correctneg=0
    for sentence, sentiment in sentencesTest.items():
        Words = re.findall(r"[\w']+", sentence)
        score=0
        for word in Words:
            if word in sentimentDictionary:
               score+=sentimentDictionary[word]
 
        total += 1

        if sentiment == "positive":
            totalpos += 1
            if score >= threshold:
                correct += 1
                correctpos += 1
                totalpospred += 1
            else:
                totalnegpred += 1
                if PRINT_ERRORS:
                    print("ERROR (positive classified as negative):", sentence)

        else:
            totalneg += 1
            if score < threshold:
                correct += 1
                correctneg += 1
                totalnegpred += 1
            else:
                totalpospred += 1
                if PRINT_ERRORS:
                    print("ERROR (negative classified as positive):", sentence)

 
    
# TODO for Step 5: Add some code here to calculate and print: (1) accuracy; (2) precision and recall for the positive class; 
# (3) precision and recall for the negative class; (4) F1 score;
    
    # accuracy
    accuracy = correct / float(total)

    # precision and recall for the positive class
    precision_positive = correctpos / float(totalpospred)
    recall_positive = correctpos / float(totalpos)

    # precision and recall for the negative class
    precision_negative = correctneg / float(totalnegpred)
    recall_negative = correctneg / float(totalneg)

    # F1 scores
    f1_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
    f1_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)
    f1_average = (f1_positive + f1_negative) / 2.0

    print("\nEvaluation Metrics (Dictionary-based):")
    print("Accuracy:", round(accuracy, 3))
    print("Precision (Positive):", round(precision_positive, 3), "Recall (Positive):", round(recall_positive, 3))
    print("Precision (Negative):", round(precision_negative, 3), "Recall (Negative):", round(recall_negative, 3))
    print("F1 Score (Averaged):", round(f1_average, 3))

######-------#######

def testDictionaryImproved(test, testLabels, sentimentDictionary):
    print("\nImproved Dictionary-based classification")

    total = len(test)
    correct = 0

    totalpos = totalneg = 0
    totalpospred = totalnegpred = 0
    correctpos = correctneg = 0

    negators = ["not", "never", "no", "n't"]
    diminishers = ["slightly", "barely", "somewhat", "a bit", "little"]

    for i in range(total):
        words = test[i].lower().split()
        score = 0
        skip = False

        for j, w in enumerate(words):

            # diminishers: cut sentiment strength by half
            if w in diminishers and j+1 < len(words):
                next_word = words[j+1]
                if next_word in sentimentDictionary:
                    score += 0.5 * sentimentDictionary[next_word]
                    skip = True
                    continue

            # negators: flip polarity of next word
            if w in negators and j+1 < len(words):
                next_word = words[j+1]
                if next_word in sentimentDictionary:
                    score -= sentimentDictionary[next_word]    # flip
                    skip = True
                    continue

            # normal dictionary scoring
            if not skip and w in sentimentDictionary:
                score += sentimentDictionary[w]

            skip = False

        # decide
        guess = "positive" if score >= 0 else "negative"
        true = testLabels[i]

        # statistics
        if true == "positive": totalpos += 1
        else: totalneg += 1

        if guess == "positive": totalpospred += 1
        else: totalnegpred += 1

        if guess == true:
            correct += 1
            if true == "positive": correctpos += 1
            else: correctneg += 1

    # --- evaluation metrics ---
    # accuracy
    accuracy = correct / float(total)

    # precision and recall for the positive class
    precision_positive = correctpos / float(totalpospred)
    recall_positive = correctpos / float(totalpos)

    # precision and recall for the negative class
    precision_negative = correctneg / float(totalnegpred)
    recall_negative = correctneg / float(totalneg)

    # F1 scores
    f1_positive = 2 * precision_positive * recall_positive / (precision_positive + recall_positive)
    f1_negative = 2 * precision_negative * recall_negative / (precision_negative + recall_negative)
    f1_average = (f1_positive + f1_negative) / 2.0

    print("Accuracy:", round(accuracy, 3))
    print("Precision (Positive):", round(precision_positive, 3), "Recall (Positive):", round(recall_positive, 3))
    print("Precision (Negative):", round(precision_negative, 3), "Recall (Negative):", round(recall_negative, 3))
    print("F1 Score (Averaged):", round(f1_average, 3))



# Print out n most useful predictors
def mostUseful(pWordPos, pWordNeg, pWord, n):
    predictPower={}
    for word in pWord:
        if pWordNeg[word]<0.0000001:
            predictPower[word] = 1000000000
        else:
            predictPower[word]=pWordPos[word] / (pWordPos[word] + pWordNeg[word])
            
    sortedPower = sorted(predictPower, key=predictPower.get)
    head, tail = sortedPower[:n], sortedPower[len(predictPower)-n:]
    print ("NEGATIVE:")
    print (head)
    print ("\nPOSITIVE:")
    print (tail)




#---------- Main Script --------------------------


sentimentDictionary={} # {} initialises a dictionary [hash function]
sentencesTrain={}
sentencesTest={}
sentencesNokia={}

#initialise datasets and dictionaries
readFiles(sentimentDictionary,sentencesTrain,sentencesTest,sentencesNokia)

pWordPos={} # p(W|Positive)
pWordNeg={} # p(W|Negative)
pWord={}    # p(W) 

# build conditional probabilities using training data
trainBayes(sentencesTrain, pWordPos, pWordNeg, pWord)

# run naive bayes classifier on datasets
# testBayes(sentencesTrain,  "Films (Train Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
testBayes(sentencesTest,  "Films  (Test Data, Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.5)
# testBayes(sentencesNokia, "Nokia   (All Data,  Naive Bayes)\t", pWordPos, pWordNeg, pWord,0.7)



# run sentiment dictionary based classifier on datasets
# testDictionary(sentencesTrain,  "Films (Train Data, Rule-Based)\t", sentimentDictionary, 1)
testDictionary(sentencesTest,  "Films  (Test Data, Rule-Based)\t",  sentimentDictionary, 1)
# testDictionary(sentencesNokia, "Nokia   (All Data, Rule-Based)\t",  sentimentDictionary, 1)

# run improved dictionary based classifier on datasets
# testDictionaryImproved(list(sentencesTrain.keys()), list(sentencesTrain.values()), sentimentDictionary)
# testDictionaryImproved(list(sentencesTest.keys()), list(sentencesTest.values()), sentimentDictionary)
# testDictionaryImproved(list(sentencesNokia.keys()), list(sentencesNokia.values()), sentimentDictionary)

#print most useful words
# mostUseful(pWordPos, pWordNeg, pWord, 100)
