# from langchain_google_genai import ChatGoogleGenerativeAI
# from pydantic import BaseModel,Field
# import os
# from dotenv import load_dotenv
# load_dotenv()


# model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
#                              google_api_key=os.getenv("GEMINI_API_KEY"))

# chat_history=[]

# while True:
#     user_imput=input("You: ")
#     chat_history.append(user_imput)
#     if(user_imput=='exit'):
#         break
#     res=model.invoke(chat_history)
#     chat_history.append(res.content)
#     print('AI: ',res.content)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load Gemini API key
load_dotenv()

# ---- Initialize LLM ----
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ---- Streamlit UI ----
st.set_page_config(page_title="🤖 AI Chat", layout="wide")
st.title("🤖 AI Chat with Gemini")
st.write("Type your messages below and get responses from the AI. Type 'exit' to stop.")

# ---- Session state to maintain chat history ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input box
user_input = st.text_input("Your message:", "")

if user_input:
    if user_input.lower() == "exit":
        st.warning("Chat ended. Refresh the page to start a new session.")
    else:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        # Get AI response
        with st.spinner("AI is typing..."):
            messages = [m["content"] for m in st.session_state.chat_history]
            response = model.invoke(messages)
            st.session_state.chat_history.append({"role": "ai", "content": response.content})

# Display chat history
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")
