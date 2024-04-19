import requests
import streamlit as st

def get_ollama_response(input_text):
    response=requests.post(
    "http://localhost:8000/summary/invoke",
    json={'input':{'text':input_text}})

    return response.json()['output']

## streamlit framework

st.title('Langchain Demo With Gemma API')
input_text = st.text_input("Write a poem on")

if input_text:
    st.write(get_ollama_response(input_text))