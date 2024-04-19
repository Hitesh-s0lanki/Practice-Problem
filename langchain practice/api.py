from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langserve import add_routes
import uvicorn
import os
from langchain_community.llms import Ollama
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="A Simple API Server"
)

## ollama gemma
model = Ollama(model = "gemma")
prompt = ChatPromptTemplate.from_template("'{text}' Summarize this text with all the information present")

add_routes(
    app,
    prompt|model,
    path="/summary"
)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

