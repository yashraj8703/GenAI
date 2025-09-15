from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv
load_dotenv()

model=ChatGoogleGenerativeAI(model='gemini-1.5-flash')

chat_history=[]

while True:
    user_imput=input("You: ")
    chat_history.append(user_imput)
    if(user_imput=='exit'):
        break
    res=model.invoke(chat_history)
    chat_history.append(res.content)
    print('AI: ',res.content)