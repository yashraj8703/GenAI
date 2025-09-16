"""
This script demonstrates a simple command-line chatbot using HuggingFace models with LangChain.
- Loads environment variables using `dotenv`.
- Initializes a HuggingFace text generation endpoint with the Mistral-7B-Instruct-v0.3 model.
- Sets up a conversational loop where user input is appended to the chat history.
- Uses `ChatHuggingFace` to generate responses based on the conversation context.
- Continues the conversation until the user types 'exit'.
Dependencies:
- langchain_huggingface
- python-dotenv
Usage:
Run the script and interact with the chatbot via the terminal.
Type 'exit' to end the conversation.
"""


from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-Instruct-v0.3',
    max_new_tokens=200,
    temperature=1.5,
    task='text-generation'
)
chat_history=[]
model=ChatHuggingFace(llm=llm)
while True:
    query=input("You: ")
    chat_history.append(query)
    if(query=='exit'):
        break
    ans=model.invoke(chat_history).content
    chat_history.append(ans)
    print(ans)



