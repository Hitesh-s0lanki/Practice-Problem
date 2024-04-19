## Pdf reader
from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('resume.pdf')
docs=loader.load()

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
documents[:5]

## Vector Embedding And Vector Store
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(documents, OpenAIEmbeddings())

query = "how many years of work experience ?"
retireved_results=db.similarity_search(query)
print(retireved_results[0].page_content)