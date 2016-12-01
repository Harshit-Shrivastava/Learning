#!/usr/bin/env python
# -*- coding: utf-8 -*- 
'''
Here I answer the requirements designated in the assignment prompt. 

(1) a description of how you formulated the problem, including precisely defining the abstractions; 
This is the text classification problem using (almost) Naive Bayes approach. I clarified the structure of Bayes net here: https://piazza.com/class/irnmu9v26th48r?cid=310
P(W_1,...W_N, D,T) = P(W_1|T)... P(W_N|T) P(T|D)P(D)
Then we just use standard approach to predict labels. When predicting, we use P(W_1,...W_N ,T)  = P(W_1|T)... P(W_N|T) P(T)  (D is eliminated). 

If some of the labels are missing, I exactly follow the idea on hint:
“To deal with missing data, adopt an iterative approach. Estimate values in the conditional probabilities tables using the portion of the data that is labeled (or randomly, if no data is labeled). Then, estimate the distribution over T for each document D using variable elimination. Now you have “hallucinated” labels for the training set and can re-estimate the conditional probability tables, which can then be used to re-estimate the topic distributions, and so on. Continue iterating until convergence or until you run out of time. :) “

(2) a brief description of how your program works; 
It just implements the idea in (1). Each conditional probabilities are computed in this way, for example:
P(W_i=apple|T=topic1) = (# of topic1 documents that have apple) / (# of topic1 documents)
Then, I just need to apply Naive Bayes approach. 

For unsupervised case I used the approach 1 suggested here: https://piazza.com/class/irnmu9v26th48r?cid=325
"Just ignore this issue and treat the topic labels as the arbitrary numbers that they are. This mean you might do a really good job of finding the topics but the numbers might not line up with the ground truth, so it will look like you're getting very bad classification results."
So the accuracy and confusion matrix are not meaningful. 

*For more implementation level details, please refer to the source code. It has a comments for each meaningful block. ***Please read the code, line by line, and their comments carefully.*** (Sorry this sounds very redundant, but last time TA missed the comments in the middle of the source, so I emphasize.)

(3) a discussion of any problems, assumptions, simplifications, and/or design decisions you made

When I have a problem, I usually ask on Piazza. Here I list the discussions that I had. Please refer to them. 

Implementation Trick to compute likelihood efficiently : https://piazza.com/class/irnmu9v26th48r?cid=338
How to compute each conditional probabilities: https://piazza.com/class/irnmu9v26th48r?cid=330 , https://piazza.com/class/irnmu9v26th48r?cid=324 
How to preprocess the document files: https://piazza.com/class/irnmu9v26th48r?cid=311
When the fraction is unreasonably low, there are not enough examples to cover all the topics in the labeled data: https://piazza.com/class/irnmu9v26th48r?cid=347
Confusion matrix computation: https://piazza.com/class/irnmu9v26th48r?cid=311
Document file name is not unique!  https://piazza.com/class/irnmu9v26th48r?cid=339

(4) answers to any questions asked below in the assignment.
We are only asked one prompt specifically. “ In your report, show the accuracies as a function of fraction; particularly interesting cases are when fraction is 1 (fully supervised), 0 (unsupervised), or low (e.g. 0.1). “ I report this point in a separated PDF file called “report-part2.pdf”. Please refer to them. 

'''

'''
notes (command examples)

python topics.py train ./train model 1.0
python topics.py test ./test model 1.0

python topics.py train ./train model 0.95
python topics.py test ./test model 0.95


#for debug
python topics.py auto ./train model1 1.0
python topics.py auto ./train model2 0.90
python topics.py auto ./train model3 0.80
python topics.py auto ./train model4 0.70
python topics.py auto ./train model5 0.60
python topics.py auto ./train model6 0.50
python topics.py auto ./train model7 0.40
python topics.py auto ./train model8 0.30
python topics.py auto ./train model9 0.20
python topics.py auto ./train model10 0.10
python topics.py auto ./train model11 0

'''

import math
import sys
import os
import operator
import string
import pickle
import random
epsilon = 0.000000000001
from collections import Counter

TOPICS=["atheism","autos","baseball","christian","crypto","electronics","forsale","graphics","guns","hockey","mac","medical","mideast","motorcycles","pc","politics","religion","space","windows","xwindows"]

def train(data_directory,model,fraction):
    vocabulary_list=set()

    print('Training... Please wait')

    #first, we will collect informations that is needed to compute conditional probabilites. 
    document_id2topic={}#document_id2topic[topic+document_id]=topic
    word2freq={}#word2freq[word]=frequency
    #collect all the words first
    #collect document-topic pairs
    #count word frequencies
    for topic in TOPICS:
        for fname in os.listdir(data_directory+'/'+topic):
            word_set_for_a_document=set()
            with open(data_directory+'/'+topic+'/'+fname,'r') as f:
                lines = f.readlines()
            if random.random() < fraction:
                document_id2topic[topic+fname]=topic
            else:
                document_id2topic[topic+fname]=0
            
            for line in lines:
                line = line.lower()
                line = line.translate(None,string.punctuation)
                for word in line.strip().split(' '):
                    if len(word)!=0:
                        if word not in word_set_for_a_document:# if already appeared in this document, skip it
                            vocabulary_list.add(word)
                            word_set_for_a_document.add(word)
                            if word not in word2freq:
                                word2freq[word]=1
                            else:
                                word2freq[word]+=1

    #if it is unsuprvised, we will assin topic for each document randomly
    if fraction == 0:
        for true_topic in TOPICS:
            for fname in os.listdir(data_directory+'/'+true_topic):
                document_id2topic[true_topic+fname] = TOPICS[random.randint(0, 19)]    

    #based on the information we collected, we will compute condtional probabilites
    prob_T,prob_W_given_T=compute_probabilities(data_directory,vocabulary_list,document_id2topic,fraction)

    #Dealing with very special cases
    #citation: https://piazza.com/class/irnmu9v26th48r?cid=347
    if len(prob_T.keys())!=20:
        #find out which topic is missing
        missing_topics=set(TOPICS)-set(prob_T.keys())

        #fix P(T) for missting topics
        for missing_topic in missing_topics:
            prob_T[missing_topic]=random.random()
        #normalize again
        prob_sum=sum(prob_T.values())
        for topic in TOPICS:
            prob_T[topic]=prob_T[topic]/prob_sum

        #fix P(W|T) for missting topics
        for missing_topic in missing_topics:
            for word in vocabulary_list:
                prob_W_given_T[missing_topic][word] = random.random()
            #normalize again
            prob_sum=sum(prob_W_given_T[missing_topic].values())
            for word in vocabulary_list:
                prob_W_given_T[missing_topic][word] = prob_W_given_T[missing_topic][word]/prob_sum

    #here we start EM process if it is not fully-supervised case
    #Design Decision: We only iterate k times. k is a small number (k<10). 
    #This is because, if we iterate until the  likelihood converge, it will take too much time. 
    #Also we observed that it will take around 5 mins for small k (k<10) so we decided to limit ourself to small k
    #Then, in order to save computation time, we do not even compute log likelihood for each iteration because we will not use it anyway. 
    k=5
    if fraction !=1:
        for i in xrange(k):
            print i
            document_id2topic_estimate=predict_train(data_directory,vocabulary_list,prob_W_given_T,prob_T,document_id2topic,fraction)
            prob_T,prob_W_given_T=compute_probabilities(data_directory,vocabulary_list,document_id2topic_estimate,fraction)
    #EM process ends.

    prob_W={}
    #prior probability of word
    #prob_W["apple"] is probability of "apple", which is just normalized frequency count
    num_total_words=sum(word2freq.values())
    for word,word_count in word2freq.iteritems():
        prob_W[word]=1.0*word_count/num_total_words

    prob_T_given_W_inverted={}
    #store P(T|W)
    #for example, prob_T_given_W_inverted['atheism']["apple"] is P(T= atheism | W = 'apple')
    #compute P(T|W) using Bayes Rule P(T|W) = P(W|T)P(T)/P(W)
    #note that order of keys are differnent than the other conditional probabilies
    for topic in TOPICS:
        prob_T_given_W_inverted[topic]={}

    for topic in TOPICS:
        for word in prob_W.keys():
            prob_T_given_W_inverted[topic][word]=prob_W_given_T[topic][word]*prob_T[topic]/prob_W[word]

    #outpout "distinctive_words.txt"
    with open("distinctive_words.txt", 'w') as f:
        for topic in prob_T.keys():
            f.write("T = %s ;top 10 words with highest P(T|W) \n"%topic)
            #the following syntax is copied from: http://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary
            top_10_words=sorted(prob_T_given_W_inverted[topic], key=prob_T_given_W_inverted[topic].get, reverse=True)[:10]
            i=1
            for word in top_10_words:
                f.write(str(i)+" : "+word+'\n')
                i+=1
            f.write("\n")

    #save the model as pickle
    with open(model, 'w') as f:
        pickle.dump((vocabulary_list,prob_W_given_T,prob_T,prob_W,prob_T_given_W_inverted),f)

    print('Training complete. The model is saved on %s'%(model))

def compute_probabilities(data_directory,vocabulary_list,document_id2topic,fraction,unsupervised=False):
    topic2word_document_count={}
    # for example, topic2word_document_count["atheism"]["apple"] is the number of atheism documents that has apple
    for topic in TOPICS:
        #initialize by topic
        topic2word_document_count[topic]={}
    
    #collect topic2word_document_count
    for topic in TOPICS:
        for fname in os.listdir(data_directory+'/'+topic):
            topic_assignment=document_id2topic[topic+fname]      
            if topic_assignment == 0:#0 means "unknown"
                continue
            word_set_for_a_document=set()
            with open(data_directory+'/'+topic+'/'+fname,'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.lower()
                line = line.translate(None,string.punctuation)
                for word in line.strip().split(' '):
                    if len(word)!=0:
                        if word not in word_set_for_a_document:# if already appeared in this document, skip it
                            word_set_for_a_document.add(word)
                            #word frequency count per topic
                            if word not in topic2word_document_count[topic_assignment]:
                                topic2word_document_count[topic_assignment][word]=1
                            else:
                                topic2word_document_count[topic_assignment][word]+=1


    prob_T={}
    #prior probability of each topic
    #prob_T["atheism"] is probability of having atheism, which is (# of documents that has topic of atheism)/(# of documents)
    topic2total_document_count = Counter(document_id2topic.values())
    # for example, topic2total_document_count["atheism"] is the number of atheism documents.
    del topic2total_document_count[0]

    num_all_document=sum(topic2total_document_count.values())
    for topic,total_count in topic2total_document_count.iteritems():
        prob_T[topic]=1.0*total_count/num_all_document
        if prob_T[topic] == 1:
            prob_T[topic]=1-epsilon
        if prob_T[topic] == 0:
            prob_T[topic]=epsilon

    prob_W_given_T={}
    #P(W|T)
    #conditonal probabilitises of word given topic
    #prob_W_given_T[topic][word]
    #for example, prob_W_given_T["atheism"]["apple"] is P(w = apple| T= atheism)
    #P(word=apple|T= atheism) = (# of atheism documents that have apple) / (# of atheism documents)
    #P(word=apple|topic1) = (# of topic1 documents that have apple) / (# of topic1 documents)
    for topic in TOPICS:
        #initialize by topic
        prob_W_given_T[topic]={}

    for word in vocabulary_list:
        for topic in TOPICS:
            if word in topic2word_document_count[topic]:
                probability=1.0*topic2word_document_count[topic][word]/topic2total_document_count[topic]
                if probability == 1:
                    probability=1-epsilon
            else:
                probability=epsilon
            prob_W_given_T[topic][word]=probability

    return prob_T,prob_W_given_T

#this is the function to use for each EM process. 
#what is does is, just predict labels of non-labeled documents, and return the prdiction.
def predict_train(data_directory,vocabulary_set,prob_W_given_T,prob_T,document_id2topic,fraction):

    document_id2topic_estimate={}

    for true_topic in TOPICS:
        for fname in os.listdir(data_directory+'/'+true_topic):
            if document_id2topic[true_topic+fname] != 0 and fraction !=0:#0 means "unknown", so if it is not unknown, we do not predict it
                document_id2topic_estimate[true_topic+fname]=true_topic
                continue
            word_set_for_a_document=set()
            with open(data_directory+'/'+true_topic+'/'+fname,'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.lower()
                line = line.translate(None,string.punctuation)
                for word in line.strip().split(' '):
                    if len(word)!=0:
                        if word not in word_set_for_a_document:# if already appeared in this document, skip it
                            word_set_for_a_document.add(word)
            prediction=predict(word_set_for_a_document,vocabulary_set,prob_W_given_T,prob_T)
            document_id2topic_estimate[true_topic+fname]=prediction

    return document_id2topic_estimate

#this is the fucntion to predict the label for a document. 
def predict(word_set_for_a_document,vocabulary_set,prob_W_given_T,prob_T):
    log_probabilites_by_topic={}
    #temporal dictonary to store probabilites by topics P(T|W_1,...,W_N)
    for topic in TOPICS:
        log_prob=math.log(prob_T[topic])
        for word in word_set_for_a_document:
            if word in vocabulary_set:
                log_prob+=math.log(prob_W_given_T[topic][word]) - math.log(1 - prob_W_given_T[topic][word])  
        log_probabilites_by_topic[topic]=log_prob

    #the following syntax is copied from: http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
    prediction=max(log_probabilites_by_topic.iteritems(), key=operator.itemgetter(1))[0]
    return prediction

def test(data_directory,model):
    print "loading pre-trained model"
    with open(model, 'r') as f:
        vocabulary_set,prob_W_given_T,prob_T,prob_W,prob_T_given_W_inverted=pickle.load(f)

    print "testing"
    truth_list=[]
    prediction_list=[]
    for true_topic in TOPICS:
        for fname in os.listdir(data_directory+'/'+true_topic):
            word_set_for_a_document=set()
            with open(data_directory+'/'+true_topic+'/'+fname,'r') as f:
                lines = f.readlines()
            for line in lines:
                line = line.lower()
                line = line.translate(None,string.punctuation)
                for word in line.strip().split(' '):
                    if len(word)!=0:
                        if word not in word_set_for_a_document:# if already appeared in this document, skip it
                            word_set_for_a_document.add(word)
            prediction=predict(word_set_for_a_document,vocabulary_set,prob_W_given_T,prob_T)
            truth_list.append(true_topic)
            prediction_list.append(prediction)

    #make confusion matrix
    #citation:https://piazza.com/class/irnmu9v26th48r?cid=340
    confusion_mat = [ [ sum( [ (truth_list[k], prediction_list[k]) == (col, row) for k in range(len(truth_list)) ] ) for row in TOPICS ] for col in TOPICS  ] 
    print "\n".join([ " ".join([ "%3d"%e for e in row]) for row in confusion_mat ] )
    print "where index is: ",TOPICS

    #compute accuracy
    correct_count=sum([confusion_mat[i][i] for i in xrange(len(confusion_mat))])
    accuracy=(1.0*correct_count/len(truth_list))
    print "accuracy is",accuracy

    return accuracy

mode,data_directory,model = sys.argv[1:3+1]
if mode == 'train':
    fraction=float(sys.argv[4])
    train(data_directory,model,fraction)
elif mode == 'test':
    test(data_directory,model)
elif mode == 'auto':#this is debug option. 
    fraction=float(sys.argv[4])
    train("./train",model,fraction)
    test("./test",model)