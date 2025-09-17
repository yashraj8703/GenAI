from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnablePassthrough,RunnableLambda,RunnableBranch

import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# IF ELSE OF LANGCHAIN WORLD




model1= ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))


positive_prompt = PromptTemplate(
    template="The following text expresses POSITIVE sentiment. Give a friendly encouraging response:\n{text}",
    input_variables=["text"]
)
negative_prompt = PromptTemplate(
    template="The following text expresses NEGATIVE sentiment. Give an empathetic and supportive response:\n{text}",
    input_variables=["text"]
)

neutral_prompt = PromptTemplate(
    template="The following text is NEUTRAL. Give a factual, calm reply:\n{text}",
    input_variables=["text"]
)

prompt = PromptTemplate(
    template="Classify the sentiment of this text as Positive, Negative, or Neutral:\n{text}",
    input_variables=["text"]
)

classifierChain=RunnableSequence(prompt,model1,StrOutputParser())

branch=RunnableBranch(
    (lambda x:x['sentiment'] == 'Positive',RunnableSequence(positive_prompt, model1,StrOutputParser())),
    (lambda x:x['sentiment'] == 'Negative',RunnableSequence(positive_prompt, model1,StrOutputParser())),
    RunnableSequence(neutral_prompt, model1, StrOutputParser())
)
chain = RunnableSequence(
    {
        "sentiment": classifierChain,
        "text": lambda x: x  # pass original input too
    },
    branch
)
result= chain.invoke("I Don't love this product!.")
print(result)