from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))

prompt1= PromptTemplate(
    template="Create a joke on topic {topic}",
    input_variables=['topic']
)

prompt2=PromptTemplate(
    template="Explain the joke : {joke}",
    input_variables=['joke']
)
parser=StrOutputParser()

RunnableSequenceChain=RunnableSequence(prompt1,model,parser,prompt2,model,parser)
# chain= prompt1|model|parser|prompt2|model|parser
result=RunnableSequenceChain.invoke({'topic':"IT Job"})
print(result)