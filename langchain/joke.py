from typing import List

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.output_parsers import OutputFixingParser
from dotenv import load_dotenv
import os

load_dotenv()


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


actor_query = "Generate the filmography for a random actor."

parser = PydanticOutputParser(pydantic_object=Actor)

misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

#define the model 
model = ChatOpenAI(model_name="gpt-3.5-turbo-instruct", temperature=0.0)

# print(parser.parse(misformatted))

new_parser = OutputFixingParser.from_llm(parser=parser, llm = ChatOpenAI())

print(new_parser.parse(misformatted))
