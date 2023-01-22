import streamlit as st
import modelBLSTM
import TFIDF
import spacy_key
import yake_key
import norm
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.header('keywords Extraction and topic modeling')

text_ab=st.text_input('Put your text here', value="", label_visibility="visible")
chek=st.button('Get keywords and topic')
if chek:

 corpus = []

 norm.normalise(corpus,text_ab)

 
 key_tf=TFIDF.tfdd_idf_keys(corpus)
 key_spacy=spacy_key.spacy_keys(corpus[0])
 key_yake=yake_key.yake_keys(corpus[0])




 
 max_len = 150
 tok=modelBLSTM.get_tok()
 sequences = tok.texts_to_sequences(corpus)
 sequences_matrix = pad_sequences(sequences,maxlen=max_len,padding='post')
 


 import tensorflow as tf 
 modelBLSTM = tf.keras.models.load_model('BLSTMmodel.hdf5')
 pred=modelBLSTM.predict(sequences_matrix)





 st.subheader('Article Keywords:')
 col1, col2, col3= st.columns(3,gap="large")
 with col1 :
   st.subheader("TF_IDF Keywords")
   for i in key_tf:
    st.write(i)
   

 with col2 :
   st.subheader("Spacy Keywords")
   for i in key_spacy:
     st.write(i)
   
 
 with col3 :
   st.subheader("Yake Keywords")
   for i in key_yake:
     st.write(i)
   
 

 if round(pred[0,0]) == 0:
     st.subheader('topic: COVID-19')
 elif round(pred[0,0])>0:
     st.subheader('topic: Non COVID-19')


 

         






      


