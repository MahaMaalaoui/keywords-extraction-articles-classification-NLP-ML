import Data
import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Embedding, Bidirectional
from tensorflow.keras.models import Sequential

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"

###################################################################################
df=Data.getdata()



max_words = 10000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(df['text'])
sequences = tok.texts_to_sequences(df['text'])
vocab_size = len(tok.word_index) + 1

sequences_matrix =pad_sequences(sequences,maxlen=max_len,padding='post')

def get_tok():
  return(tok)
##################################################################################"

#from sklearn.model_selection import train_test_split
#x_train,x_test,train_label,test_label=train_test_split(sequences_matrix,df['labels'],test_size=0.33,random_state=0)





                                                        

#modelBLSTM= Sequential()   
#modelBLSTM.add(Embedding(vocab_size,150,input_length=max_len))
#modelBLSTM.add(Bidirectional(LSTM(150,dropout=0.5)))
#modelBLSTM.add(Dense(1,activation='sigmoid'))

#modelBLSTM.compile('adam',loss='binary_crossentropy', metrics=['accuracy'])

#modelBLSTM.fit(x_train,train_label,batch_size=500,epochs=25,validation_split=0.2)


#modelBLSTM.save('C:/Users/Maha/Desktop/projetnlp/BLSTMmodel.hdf5')



