import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/sms spam classifier")
input_sms = st.text_area("Enter the message")
if st.button('predict'):
    #1.preprocess
    def transform_text(text):
        text = text.lower()
        text = nltk.word_tokenize(text)
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
        text = y[:]#cloning list
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(i)
        text = y[:]
        y.clear()

        for i in text:
            y.append(ps.stem(i))
        return " ".join(y)

    transformed_sms = transform_text(input_sms)
    #2.vectorrize
    vector_input = tfidf.transform([transformed_sms])
    #3.predict
    result = model.predict(vector_input)
    #4.display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")