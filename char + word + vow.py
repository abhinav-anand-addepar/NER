# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 00:52:06 2019

@author: abhinav
"""

import pandas as pd
import numpy as np

from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
 
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
max_len = 0 


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

train = readfile("data/train.txt")
test = readfile("data/test.txt")
valid = readfile("data/valid.txt")
         


def addchar( fulldataset ):
    for i, dataset in enumerate(fulldataset):
        for j, data in enumerate(dataset):
            chars = [c for c in data[0]]
            fulldataset[i][j] = [data[0], chars, data[1]]
    return  fulldataset 

train = addchar(train)
test = addchar(test)
valid = addchar(valid)
st =   set() 


for  j  in train:
    for i in j :      
        if i[2] == "O\n":
             
            st.add( i[0].lower() ) 
        
    

maxvowf  = 0 
def addvow(  fulldataset ):
    maxvow = 0 
    for i ,dataset in enumerate( fulldataset ): 
        for j,data in enumerate( dataset ) :
            ch = data[1]
            count = 0
            
            for  k  in ch:
                if k.lower()  in ['a','e','i','o','u' ]:
                    count = count + 1
            if data[0].lower() in st:
                count = 0 
            
            maxvow = max( maxvow , count  )
            fulldataset[i][j] = [data[0], data[1],count,data[2] ]
    return fulldataset , maxvow
                    
train ,maxvow1 = addvow( train )
test  , maxvow2 = addvow( test )

valid ,maxvow3 = addvow( valid ) 
   
maxvowf = max( maxvow1 , maxvow2 , maxvow3 )         

labels = set()
words = {}

for data in [train,test,valid]:
    for sentence in data:
        for word, char, vow,  token in sentence:
            labels.add(token)
            words[word.lower()] = True
            

label2Idx = {}
for label in labels:
    label2Idx[label] = len(label2Idx)




word2Idx = {}
wordEmbeddings  = [] 


f = open( "embeddings/glove.6B.100d.txt" , encoding="utf-8" )
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
char2Idx = {"PADDING":0, "UNKNOWN":1}
for c in " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|":
    char2Idx[c] = len(char2Idx)



vow2vec = np.eye(  maxvowf +1  )

def creatematrix(sentences, wordid, labelid, charid , vow2vec  ):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    
    for sentence in sentences:
        wordIndices = []    
   
        charIndices = []
        labelIndices = []
        vowidx=[] 
        for word,char, vow , label in sentence:  
            wordCount += 1
            if word in wordid:
                wordIdx = wordid[word]
            elif word.lower() in wordid:
                wordIdx = wordid[word.lower()]                 
            else:
                wordIdx = unknownIdx
                unknownWordCount += 1
            charIdx = []
            for x in char:
                charIdx.append(charid[x.lower()])
            #Get the label and map to int            
            wordIndices.append(wordIdx)
       
            charIndices.append(charIdx)
            labelIndices.append(labelid[label])
            vowidx.append( vow )
           
        dataset.append([wordIndices, charIndices, vowidx,   labelIndices]) 
        
    return dataset
        


trainf = creatematrix(train , word2Idx ,  label2Idx , char2Idx  , vow2vec   )
testf = creatematrix(test , word2Idx ,  label2Idx  , char2Idx    , vow2vec    )
validf = creatematrix(valid  , word2Idx ,  label2Idx   , char2Idx    , vow2vec   )



def pad(Sentences):
    maxlen = 52
    for sentence in Sentences:
        char = sentence[1]
        maxlen = max(maxlen,len( char ))
            
    for i,sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1],52,padding='post')
    return Sentences

trainf = pad(trainf)
testf = pad(testf )
validf = pad(validf )

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
        
        char = [] 
    
        labels = []
        
        v = [] 
        data = dataset[start:i]
        start = i
        for dt in data:
            t,  ch,   vv , l = dt
            l = np.expand_dims(l, -1)
            tokens.append(t)
            
            v.append( vv )
            char.append(ch )
            labels.append(l)
        
        yield np.asarray(labels),np.asarray(char),          np.asarray( v  )      ,       np.asarray(tokens) 
        

train_batch,train_batch_len = createBatches(trainf )
valid_batch, valid_batch_len = createBatches(validf )
test_batch,test_batch_len = createBatches(testf )

words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)

vow_input = Input(shape=(None,   ),dtype='int32',name='vow_input')
vows  = Embedding(input_dim=vow2vec.shape[0], output_dim=vow2vec.shape[1],  weights=[ vow2vec] )(vow_input)





character_input=Input(shape=(None,52 ),name='char_input')
embed_char_out=TimeDistributed(Embedding(len(char2Idx),30,embeddings_initializer=RandomUniform(minval=-0.5, maxval=0.5)), name='char_embedding')(character_input)
dropout= Dropout(0.5)(embed_char_out)
conv1d_out= TimeDistributed(Conv1D(kernel_size=3, filters=30, padding='same',activation='tanh', strides=1))(dropout)
maxpool_out=TimeDistributed(MaxPooling1D(52))(conv1d_out)
char = TimeDistributed(Flatten())(maxpool_out)
char = Dropout(0.5)(char)
output = concatenate([words,  char , vows  ])
output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(output)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs=[words_input,  character_input , vow_input  ], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
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
        tokens, char,  vo ,   labels = data
        tokens = np.asarray([tokens])     
        char = np.asarray([char])
        
        vo = np.asarray( [vo ] )
        
        pred = model.predict([tokens , char  , vo  ], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    b.update(i+1)
    return predLabels, correctLabels




epochs=  20  
for epoch in range(epochs):    
    print("Epoch %d/%d"%(epoch,epochs))
    a = Progbar(len(train_batch_len))
    for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
        labels,  char,  vow , tokens  = batch       
        model.train_on_batch([tokens               ,char , vow ],  labels)
        a.update(i)
        
    predLabels, correctLabels = tag_dataset(valid_batch)        
    pre_dev, rec_dev, f1_dev = compute_f1(predLabels, correctLabels, idx2Label)
    print("Dev-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_dev, rec_dev, f1_dev))
    
#   Performance on test dataset       
    predLabels, correctLabels = tag_dataset(test_batch)        
    pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))
    a.update(i+1)
    print(' ')




def creatematrixff(sentences, wordid, labelid, charid ):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
        
    dataset = []
    
    wordCount = 0
    unknownWordCount = 0
    

    wordIndices = []    
   
    charIndices = []
    labelIndices = []
        
    for word,char,label in sentences:  
      
        wordCount += 1
        if word in wordid:
            wordIdx = wordid[word]
        elif word.lower() in wordid:
            wordIdx = wordid[word.lower()]                 
        else:
            wordIdx = unknownIdx
            unknownWordCount += 1
        charIdx = []
        for x in char:
            charIdx.append(charid[x.lower() ])
            #Get the label and map to int            
        wordIndices.append(wordIdx)
       
        charIndices.append(charIdx)
        labelIndices.append(labelid[label])
           
    dataset.append([wordIndices, charIndices, labelIndices]) 

    return dataset 

def addcharff( fulldataset ):
    
    for j, data in enumerate(fulldataset):
        chars = [c for c in data[0]]
        fulldataset[j] = [data[0], chars, data[1]]
    return  fulldataset 

def padff(Sentences):
    maxlen = 52
 
            
    for i,sentence in enumerate(Sentences):
        Sentences[i][1] = pad_sequences(Sentences[i][1],52,padding='post')
    return Sentences

def predictff(sentence ):
    mm= [] 
    
    splits = sentence.split(' ')
    for i in splits:
        mm.append( [ i , 'O\n'   ] )
    t=mm 
    
    mm = addcharff(mm)
      
    mm=creatematrixff( mm ,word2Idx ,     label2Idx   , char2Idx  )
                   
    mm = padff(mm) 
    
    predLabels, correctLabels = tag_dataset( mm )
    tag = []
    for i in     range(len( predLabels[0]  )):
        tag.append(idx2Label[predLabels[0][i]  ] ) 
    print(tag) 
    ans = []
    for i in range(len(tag )):
        ans.append([ t[i][0]     ,          tag[i]]  )
    return ans 
    
    
    