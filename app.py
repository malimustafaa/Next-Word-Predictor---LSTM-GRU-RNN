import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#load lstm model
model = load_model('next_word_lstm.h5')

#load the tokenizer
with open('tokenizer.h5','rb') as file:
    tokenizer = pickle.load(file)

#def predict function

def predict_next_word(model,tokenizer,text,max_sequence_len):
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = pad_sequences([token_list],maxlen=max_sequence_len-1,padding='pre')
    predicted = model.predict(token_list,verbose=0)
    predicted_word_index = np.argmax(predicted,axis=1) #returns index of max probability
    for word,index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

#streamlit app
st.title("Next Word Predictor with LSTM and Early Stopping")
input_text = st.text_input("Enter the sequence of words")
if st.button("predict next word"):
    max_seq_len = model.input_shape[1] + 1
    next_word = predict_next_word(model,tokenizer,input_text,max_seq_len)
    st.write(f"Next word: {next_word}")


