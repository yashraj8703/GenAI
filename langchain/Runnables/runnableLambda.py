from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


#                       passthrough to print joke
# prompt->model->parser->
#                       runnable lambda to count no of words

model1= ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))


prompt1=PromptTemplate(
    template="Create one very funny joke on : {topic} and do not add any markups",
    input_variables=['topic']
)

def WordCounter(text):
    return len(text.split())


runnableWordCounter=RunnableLambda(WordCounter)

parser = StrOutputParser()
jokeGeneratorChain=RunnableSequence(prompt1,model1,parser)

parallelChain=RunnableParallel(
    {
        "joke":RunnablePassthrough(),
        "word_count":runnableWordCounter
    }
)


chain=jokeGeneratorChain|parallelChain


result= chain.invoke({"topic":"cricket"})
print(result)
