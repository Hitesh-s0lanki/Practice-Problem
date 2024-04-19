from typing import List
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_ai21 import AI21SemanticTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI, OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain_openai import OpenAI
from langchain.output_parsers import OutputFixingParser
import os

from htmlTemplate import css, bot_template, user_template
import re 
import datetime

def get_pdf_text(pdf_docs):
    text = ""
    i = 1
    for pdf in pdf_docs:
        
        text += "Candidate-" + str(i) + "\t" 
        
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
            
    text = text.lower().strip()
    current_year = datetime.datetime.now().year
    text = text.replace('present', str(current_year))

    pattern = r'[^a-z0-9/+\-.@]'

    # Find all matches
    new_text = re.sub(pattern, ' ', text)
    
    return new_text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def retrive_resume_candidate_data(context):
    class CandidateDetails(BaseModel):
        name: str = Field(description="Name of the candidate")
        email: str = Field(description="Email of the candidate")
        total_year_experience : float = Field(description="Total year of work experience")

    class ListCandidate(BaseModel):
        candidates: List[CandidateDetails] =  Field("List of all Candidate data")
        

    # Set up a parser + inject instructions into the prompt template.
    parser = JsonOutputParser(pydantic_object=ListCandidate)

    prompt = PromptTemplate(
        template="""
            Answer the following question based only on the provided context. 
            Think step by step before providing a detailed answer.  
            <context>
            {context}
            </context>
            Question: {input}""",
        input_variables=["input", "context"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    response = st.session_state.conversation({'question': "Answer the following question based only on the provided context. Extract the name,email,total number of work experience of each candidates from the provided contexts.If experience is not there then 0, don't include the education year"})
    
    new_parser = OutputFixingParser.from_llm(parser=parser, llm = OpenAI())

    model = OpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

    # And a query intended to prompt a language model to populate the data structure.
    prompt_and_model = prompt | model | new_parser

    # Invoke the prompt template and model
    output = prompt_and_model.invoke({"context": context, "input": "Extract the name,email,total number of work experience of each candidates in list of Json.If experience is not there then 0, don't include the education year"})

    print(output['candidates'])

    
    


def main():
    load_dotenv()
    
    os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
    os.environ["AI21_API_KEY"] = os.getenv("AI21_API_KEY")
    
    st.set_page_config(page_title = "Chat with multiple pdfs", page_icon = ":books:")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with multiple pdfs :books:")
    user_question = st.text_input("Ask a question about your document")
    
    
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Uplod your document here and click on  'Process'", accept_multiple_files=True)

        raw_docs = ""

        if st.button("Process"):
            with st.spinner("Processing"):
                # get the text from pdf
                raw_docs = get_pdf_text(pdf_docs)
                
                # get the text chunks
                doc_chunks = get_text_chunks(raw_docs)
                
                # create vectorstore
                vectorstore = get_vectorstore(doc_chunks)
                
                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)

        # error
        # if st.button("Get Data"):
        #     retrive_resume_candidate_data(raw_docs)
                
                
    

if __name__ == "__main__":
    main()