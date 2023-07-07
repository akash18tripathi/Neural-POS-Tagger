# ====================================== ** Uncomment below 3 lines for Running in Google Colab ** ===========================================
# !curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4923{/ud-treebanks-v2.11.tgz}
# !tar -xvf  'ud-treebanks-v2.11.tgz'
# !pip install conllu

from conllu import parse
import keras
from keras.utils import to_categorical
from keras import Sequential
from keras.layers import Embedding, Dense, TimeDistributed, LSTM, GRU, SimpleRNN
import matplotlib.pyplot as plt
import numpy as np
import nltk
import string
from sklearn.metrics import classification_report
from keras.models import load_model


#====================================== ** Paths ** ======================================================
trainPath = "ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-train.conllu"
testPath = "ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-test.conllu"
devPath = "ud-treebanks-v2.11/UD_English-Atis/en_atis-ud-dev.conllu"

#============================== ** Padding length for our Model ** ========================================
maxPaddedLen = 50 
embedding_size=100

#================================== ** Vocab builder class ** =============================================
class Vocab_Builder:
  def __init__(self):
    self.vocab=set()
    self.vocab.add('<UNK>')
    self.vocab_size=1
    self.out_of_vocab_token=1
    self.d={'<UNK>':self.out_of_vocab_token}
    self.revD={self.out_of_vocab_token:'<UNK>'}
  
  def build_vocab(self,X):
    count=2
    for sent in X:
      for word in sent:
        self.vocab.add(word)
        if word not in self.d:
          self.d[word]= count
          self.revD[count]=word
          count+=1
    self.vocab_size = len(self.vocab)
    return

  def get_sequence_from_texts(self,X):
    li=[]
    for sent in X:
      temp=[]
      for word in sent:
        if word not in self.d:
          temp.append(self.out_of_vocab_token)
        else:
          temp.append(self.d[word])
      li.append(temp)
    return li

#================================== ** Padding function ** ================================================
def padding(X,maxPaddingLength):
  li=[]
  for sent in X:
    if(len(sent)>maxPaddingLength):
      li.append(sent[:maxPaddingLength])
    elif(len(sent)<maxPaddingLength):
      temp=[0]*(maxPaddingLength-len(sent))
      li.append(temp+sent)
    else:
      li.append(sent)
  return np.asarray(li)

def argmax(arr):
  n = arr.shape[0]
  index=-1
  maxVal=-float('inf')
  for i in range(1,n):
    if(maxVal<arr[i]):
      index=i
      maxVal=arr[i]
  return index

#==================================== ** Preprocessing a given sentence ** ================================================
def preprocess(sentence):
    text = sentence
    text = text.lower()
    text_p = "".join([char for char in text if char not in string.punctuation])
    words = text_p.split()
    return words


#==================================== ** Model training Code ** ================================================

#LSTM
def lstm_model_train(X_train,Y_train):
    num_of_classes=Y_train.shape[2]
    rnn = Sequential()
    rnn.add(Embedding(input_dim=vocab_tokenizer.vocab_size+1,output_dim=30, input_length=maxPaddedLen))
    rnn.add(LSTM(64,return_sequences=True))
    rnn.add(TimeDistributed(Dense(num_of_classes,activation='softmax')))
    rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    rnn.summary()

    rnn_train = rnn.fit(X_train,Y_train,batch_size=128,epochs=20,validation_data=(X_dev,Y_dev))
    rnn.save('models/lstm_model.h5')
    plt.plot(rnn_train.history['acc'])
    plt.plot(rnn_train.history['val_acc'])
    plt.title('model accuracy')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc="lower right")
    plt.show()
    return 

#GRU
def gru_model_train(X_train,Y_train):
    num_of_classes=Y_train.shape[2]
    rnn = Sequential()
    rnn.add(Embedding(input_dim=vocab_tokenizer.vocab_size+1,output_dim=30, input_length=maxPaddedLen))
    rnn.add(GRU(64,return_sequences=True))
    rnn.add(TimeDistributed(Dense(num_of_classes,activation='softmax')))
    rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    rnn.summary()

    rnn_train = rnn.fit(X_train,Y_train,batch_size=128,epochs=20,validation_data=(X_dev,Y_dev))
    rnn.save('models/gru_model.h5')
    plt.plot(rnn_train.history['acc'])
    plt.plot(rnn_train.history['val_acc'])
    plt.title('model accuracy')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc="lower right")
    plt.show()
    return 

#RNN
def simple_rnn_model_train(X_train,Y_train):
    num_of_classes=Y_train.shape[2]
    rnn = Sequential()
    rnn.add(Embedding(input_dim=vocab_tokenizer.vocab_size+1,output_dim=30, input_length=maxPaddedLen))
    rnn.add(SimpleRNN(64,return_sequences=True))
    rnn.add(TimeDistributed(Dense(num_of_classes,activation='softmax')))
    rnn.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])
    rnn.summary()

    rnn_train = rnn.fit(X_train,Y_train,batch_size=128,epochs=20,validation_data=(X_dev,Y_dev))
    rnn.save('models/simple_rnn_model.h5')
    plt.plot(rnn_train.history['acc'])
    plt.plot(rnn_train.history['val_acc'])
    plt.title('model accuracy')
    plt.grid()
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc="lower right")
    plt.show()
    return 




# ======================================= ** Train Data Preprocessing =======================================
f = open(trainPath)
trainData = f.read()
f.close()
sentences = parse(trainData)
X=[]
Y=[]
for sentence in sentences:
  x_temp=[]
  y_temp=[]
  for i in range(len(sentence)):
    x_temp.append(sentence[i]['form'])
    y_temp.append(sentence[i]['upos'])
  X.append(x_temp)
  Y.append(y_temp)

vocab_tokenizer = Vocab_Builder()
vocab_tokenizer.build_vocab(X)
encoded_X = vocab_tokenizer.get_sequence_from_texts(X)

tag_tokenizer = Vocab_Builder()
tag_tokenizer.build_vocab(Y)
encoded_Y = tag_tokenizer.get_sequence_from_texts(Y)

X_train = padding(encoded_X,maxPaddedLen)
Y_train = padding(encoded_Y,maxPaddedLen)
Y_train = to_categorical(Y_train)

print("X_train shape: ",X_train.shape)
print("Y_train shape: ",Y_train.shape)


# ================================== ** Test Data Preprocessing ** ========================================

f = open(testPath)
testData = f.read()
f.close()
sentences = parse(testData)

X=[]
Y=[]
for sentence in sentences:
  x_temp=[]
  y_temp=[]
  for i in range(len(sentence)):
    x_temp.append(sentence[i]['form'])
    y_temp.append(sentence[i]['upos'])
  X.append(x_temp)
  Y.append(y_temp)

encoded_X = vocab_tokenizer.get_sequence_from_texts(X)
encoded_Y = tag_tokenizer.get_sequence_from_texts(Y)

X_test = padding(encoded_X,maxPaddedLen)
Y_test = padding(encoded_Y,maxPaddedLen)
Y_test = to_categorical(Y_test)

print("X_test shape: ",X_test.shape)
print("Y_test shape: ",Y_test.shape)


#================================== ** Validation Data Preprocessing ** =========================================

f = open(devPath)
devPath = f.read()
f.close()
sentences = parse(devPath)

X=[]
Y=[]
for sentence in sentences:
  x_temp=[]
  y_temp=[]
  for i in range(len(sentence)):
    x_temp.append(sentence[i]['form'])
    y_temp.append(sentence[i]['upos'])
  X.append(x_temp)
  Y.append(y_temp)

encoded_X = vocab_tokenizer.get_sequence_from_texts(X)
encoded_Y = tag_tokenizer.get_sequence_from_texts(Y)
X_dev = padding(encoded_X,maxPaddedLen)
Y_dev = padding(encoded_Y,maxPaddedLen)
Y_dev = to_categorical(Y_dev)


print("X_dev shape: ",X_dev.shape)
print("Y_dev shape: ",Y_dev.shape)

#==================================== ** Model Training Part ** ============================================

# Uncomment below 3 lines to train new model
# simple_rnn_model_train(X_train,Y_train)
# lstm_model_train(X_train,Y_train)
# gru_model_train(X_train,Y_train)

#give model name which u want to load
rnn = load_model('models/lstm_model.h5')

loss, accuracy = rnn.evaluate(X_test, Y_test)
print("Loss: {0},\nAccuracy on Test Dataset: {1}%".format(loss, accuracy*100))
predictions = rnn.predict(X_test)

y_pred=[]
for i in range(predictions.shape[0]):
  li=[]
  for j in range(predictions[i].shape[0]):
    arr=[0]*predictions[i][j].shape[0]
    arr[np.argmax(predictions[i][j])]=1
    li.append(arr)
  y_pred.append(np.asarray(li))
y_pred = np.asarray(y_pred)


targets=['PADDING']
for i in tag_tokenizer.revD:
  targets.append(tag_tokenizer.revD[i])
print(targets)

# ======================= Print F1 score, precision and recall =================================
print(classification_report(np.reshape(Y_test,(Y_test.shape[0]*Y_test.shape[1],Y_test.shape[2])), np.reshape(y_pred,(y_pred.shape[0]*y_pred.shape[1],y_pred.shape[2])),target_names=targets))


# ============================ Predicting a sentence ===========================================
print("Enter a sentence: ")
sentence = str(input())
X = preprocess(sentence)
t = len(X)
encoded_X = vocab_tokenizer.get_sequence_from_texts([X])
X = padding(encoded_X,maxPaddedLen)
y = rnn.predict(X)
for i in range(maxPaddedLen-t,maxPaddedLen):
  print(vocab_tokenizer.revD[X[0][i]] +'\t'+tag_tokenizer.revD[argmax(y[0][i])])