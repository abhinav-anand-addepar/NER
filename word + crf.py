# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 19:47:36 2019

@author: abhinav
"""

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
 
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform



def readfile(name):
    file = open( name )
    ss = []
    s = []
    for line in file:
        if len(line) == 0 or line.startswith('-DOCSTART') or line[0] == "\n":
            if len(s) > 0: 
                ss.append(s)
                s = []
            continue
        
        splits  = line.split(' ')
        s.append([splits[0],splits[-1]])
        
    if len(s) > 0:
        ss.append(s)
        s=[]
    return ss

train = readfile("/content/drive/My Drive/Myproject/My ner/data/train.txt")
test = readfile("/content/drive/My Drive/Myproject/My ner/data/test.txt")
valid = readfile("/content/drive/My Drive/Myproject/My ner/data/valid.txt")
 
 

labels = set()
words = {}

for data in [train,test,valid]:
    for sentence in data:
        for word, token in sentence:
            labels.add(token)
            words[word.lower()] = True
            

label2Idx = {}
for label in labels:
    label2Idx[label] = len(label2Idx)




word2Idx = {}
wordEmbeddings  = [] 


f = open( "/content/drive/My Drive/Myproject/My ner/embeddings/glove.6B.100d.txt" , encoding="utf-8" )

for line in f:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)


def creatematrix(data,wordid , labelid):
    unknownid  = wordid["UNKNOWN_TOKEN"]
    new_data = []
    
    
    
    for sentence in data:
        wordidxfinal = []
        labelidxfinal  = []
        for word,label in sentence:
            if word in wordid:
                wordidx = wordid[word]
            elif word.lower() in wordid:
                wordidx = wordid[word.lower()]
            else:
                wordidx = unknownid
            wordidxfinal.append(wordidx)
            labelidxfinal.append(labelid[label ])
       
        new_data.append([wordidxfinal, labelidxfinal ])
        
    
    return new_data 
        


trainf = creatematrix(train , word2Idx , label2Idx   )
testf = creatematrix(test , word2Idx , label2Idx   )
validf = creatematrix(valid  , word2Idx , label2Idx   )

idx2Label = {v: k for k, v in label2Idx.items()}


def createBatches(data):
    l = []
    for i in data:
        l.append(len(i[0]))
    l = set(l)
    batches = []
    batch_len = []
    z = 0
    for i in l:
        for batch in data:
            if len(batch[0]) == i:
                batches.append(batch)
                z += 1
        batch_len.append(z)
    return batches,batch_len

def iterate_minibatches(dataset, batch_len):
    start = 0
    for i in batch_len:
        tokens = []
  
        labels = []
        data = dataset[start:i]
        start = i
        for dt in data:
            t,  l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
    
            labels.append(l)
        
        yield np.asarray(labels), np.asarray(tokens) 
        

train_batch,train_batch_len = createBatches(trainf )
valid_batch, valid_batch_len = createBatches(validf )
test_batch,test_batch_len = createBatches(testf )

from keras_contrib.layers import  CRF 
 
words_input = Input(shape=(  None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
output = Bidirectional(LSTM(200, return_sequences=True , dropout=0.50, recurrent_dropout=0.25))(words)

outputx  = Bidirectional(LSTM(200, return_sequences=True , dropout=0.50, recurrent_dropout=0.25))( output )


 


outputxx  = TimeDistributed(Dense(50, activation="relu"))(  outputx  )  # a dense layer as suggested by neuralNer
crf =            CRF( 9  ) # CRF layer
out = crf( outputxx  )


model = Model(inputs=  words_input   , outputs=  out                )
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy]) 
model.summary()



def compute_f1(predictions, correct, idx2Label):
    label_pred = []
    for sentence in predictions:
        label_pred.append([idx2Label[element] for element in sentence])

    label_correct = []
    for sentence in correct:
        label_correct.append([idx2Label[element] for element in sentence])

    # print("predictions ", len(label_pred))
    # print("correct labels ", len(label_correct))

    prec = compute_precision(label_pred, label_correct)
    rec = compute_precision(label_correct, label_pred)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec);

    return prec, rec, f1


def compute_precision(guessed_sentences, correct_sentences):
    assert (len(guessed_sentences) == len(correct_sentences))
    correctCount = 0
    count = 0

    for sentenceIdx in range(len(guessed_sentences)):
        guessed = guessed_sentences[sentenceIdx]
        correct = correct_sentences[sentenceIdx]
        assert (len(guessed) == len(correct))
        idx = 0
        while idx < len(guessed):
            if guessed[idx][0] == 'B':  # a new chunk starts
                count += 1

                if guessed[idx] == correct[idx]:  # first prediction correct
                    idx += 1
                    correctlyFound = True

                    while idx < len(guessed) and guessed[idx][0] == 'I':  # scan entire chunk
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False 

                        idx += 1

                    if idx < len(guessed):
                        if correct[idx][0] == 'I':  # chunk in correct was longer
                            correctlyFound = False

                    if correctlyFound:
                        correctCount += 1
                else:
                    idx += 1
            else:
                idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision



def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens,  labels = data
        tokens = np.asarray([tokens])     
          
        pred = model.predict([tokens ], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    return predLabels, correctLabels




epochs=   100   
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels, tokens  = batch      
        
        labels  = [to_categorical(i, num_classes= 9 ) for i in  labels ]
        
        
        model.train_on_batch([tokens ],  np.array( labels ) ) 
        a.update(i)
        
    
    if epoch %9 !=0:
      continue 
    predLabels, correctLabels = tag_dataset(valid_batch)        
    pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
    print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
#   Performance on test dataset       
    predLabels, correctLabels = tag_dataset(test_batch)        
    pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
    a.update(i+1)
    print(' ')




def creatematrixff(data,wordid , labelid):
    unknownid  = wordid["UNKNOWN_TOKEN"]
    new_data = []
    
    
    

    wordidxfinal = []
    labelidxfinal  = [] 
    
    for i  in   range(len(data) ) :
        
        word = data[i][0]
        
        label = data[i][1]
        
        
        if word in wordid:
            wordidx = wordid[word]
        elif word.lower() in wordid:
            wordidx = wordid[word.lower()]
        else:
            wordidx = unknownid
        wordidxfinal.append(wordidx)
        labelidxfinal.append(labelid[label ])
       
    new_data.append([wordidxfinal, labelidxfinal ])
        
    
    return new_data 

def predictff(sentence ):
    mm= [] 
    
    splits = sentence.split(' ')
    for i in splits:
        mm.append( [ i , 'O\n'   ] )
    
    print(mm) 
    t=mm 
    mm=creatematrixff( mm ,word2Idx ,     label2Idx   )
    predLabels, correctLabels = tag_dataset( mm )
    
    tag = []
    
    
    
    for i in     range(len( predLabels[0]  )):
         
        tag.append(idx2Label[predLabels[0][i]  ] ) 
        
        
    print(tag) 
    
    
    ans = []
    
    for i in range(len(tag )):
        ans.append([ t[i][0]     ,          tag[i]]  )
        
    return ans 
    
    
    