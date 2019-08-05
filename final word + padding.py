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

train = readfile("data/train.txt")
test = readfile("data/test.txt")
valid = readfile("data/valid.txt")
 

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
    
    
xtrain = [ [word[0]    for word in sentence] for sentence in train  ]
xtest  = [ [word[0]    for word in sentence] for sentence in  test   ]
xvalid  = [ [word[0]    for word in sentence] for sentence in     valid  ]


max_len = 0

for dataset in [xtrain,xtest , xvalid]:
    for data in dataset:
        max_len = max( max_len , len(data ))
        
   

idx2Label = {v: k for k, v in label2Idx.items()}



     
def pad(dataset ):
    new_dataset = []
    for seq in  dataset :
        new_seq = []
        for i in range(max_len):
            try:
                new_seq.append(seq[i])
            except:
                new_seq.append("PADDING_TOKEN")
        new_dataset.append(new_seq)
    return new_dataset 
 

xxtrain       = pad(xtrain) 
xxtest = pad(xtest)
xxvalid = pad(xvalid )

y = [[label2Idx[w[ 1 ]] for w in s] for s in  train ]
 
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=label2Idx["O\n"])


batch_size = 32



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
 
from keras.layers import             Lambda
from keras.layers.merge import add
words_input = Input(shape=(None,),dtype='int32',name='words_input')
words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1],  weights=[wordEmbeddings], trainable=False)(words_input)
output = Bidirectional(LSTM(200, return_sequences=True , dropout=0.50, recurrent_dropout=0.25))(words)
output = TimeDistributed(Dense(len(label2Idx), activation='softmax'))(output)
model = Model(inputs= [words_input ], outputs=[output])
model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam')
model.summary()







def creatematrix(data,wordid      ):
    unknownid  = wordid["UNKNOWN_TOKEN"]
    new_data = []
    for sentence in data:
        wordidxfinal = []
    
        for word                  in sentence:
            if word in wordid:
                wordidx = wordid[word]
            elif word.lower() in wordid:
                wordidx = wordid[word.lower()]
            else:
                wordidx = unknownid
            wordidxfinal.append(wordidx)
        new_data.append(       wordidxfinal   )
    return new_data 
        

y = np.expand_dims( y , -1)

xxftrain = creatematrix(xxtrain, word2Idx )
history = model.fit(   [xxftrain ]    , y  ,                     
                    batch_size=   32  , epochs= 1 , verbose=1)
 




def createff(data,wordid      ):
    unknownid  = wordid["UNKNOWN_TOKEN"]
    new_data = []
    wordidxfinal = []
    
    for word                  in data[0]  :
        if word in wordid :
            wordidx = wordid[word]
        elif word.lower() in wordid:
            wordidx = wordid[word.lower()]
        else:
            wordidx = unknownid
        wordidxfinal.append(wordidx)
    new_data.append(       wordidxfinal   )
    return new_data 


  
        
def padff(dataset ):
    new_dataset = []

    new_seq = []
    for i in range(max_len):
        try:
            new_seq.append(dataset[i])
        except:
            new_seq.append("PADDING_TOKEN")
    new_dataset.append(new_seq)
    return new_dataset
 
def tag_dataset(dataset):
    correctLabels = []
    predLabels = []
 
    for i,data in enumerate(dataset):    
        tokens = data
        tokens = np.asarray([tokens])     
          
        pred = model.predict([tokens ], verbose=False)[0]   
        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
 
    return predLabels, correctLabels


def predict( sentence ):
    mm= [] 
    
    splits = sentence.split(' ')
     
    for i in splits:
        mm.append( [ i , "O\n"    ] )
        
    
    t=mm 
 
    x=[] 
    y= [] 
    for i in mm : 
        x.append ( i[0]  ) 
        y.append(i[1] )
    
    
    xt = padff( x )
    
    xt = createff(xt  , word2Idx ) 
    
 
    
    
    predLabels, correctLabels = tag_dataset(  xt  )
    
    tag = []
    
    
    
    for i in     range(len( predLabels[0]  )):
         
        tag.append(idx2Label[predLabels[0][i]  ] ) 
    
   
    ans = []
    for i in range(len(t  )):
        ans.append([ t[i][0]     ,          tag[i]]  )
        
    return ans 
    



idx2Label = {v: k for k, v in label2Idx.items()}


    

predict("My name is tom and i live in usa")

