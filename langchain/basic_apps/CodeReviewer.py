import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class CodeAnalysis(BaseModel):
    problemStatement:str=Field(description="Give the problem statement")
    language: str = Field(description="Programming language of the given code")
    functionality: str = Field(description="What the code does in simple terms")
    complexity_level: str = Field(
        description="""You are a code analysis assistant. Your task is to analyze the given code and **only** provide the **Time and Space complexity** in Big O notation. 
Do not provide any explanations, reasoning, or extra text.

Code:
{code}

Provide output in the following format:
Time: <Time Complexity in Big O>
Space: <Space Complexity in Big O>"""
    )
    summary: str = Field(description="A concise explanation of the code's purpose and flow")


parser = PydanticOutputParser(pydantic_object=CodeAnalysis)

template = PromptTemplate(
    template="""You are a code analysis assistant. 
Analyze the given code snippet and provide structured details.

Code:
{code}

{format_instructions}
""",
    input_variables=["code"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# ---- LLM ----
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

model_chain = template | model | parser

# ---- Streamlit UI ----
st.set_page_config(page_title="🧠 Code Explainer Bot", layout="centered")
st.title("🧠 AI Code Explainer")
st.write("Paste your code snippet below and get a structured explanation.")

code_input = st.text_area("Enter Code Here", height=300)

if st.button("Explain Code"):
    if code_input.strip():
        with st.spinner("Analyzing code..."):
            try:
                result = model_chain.invoke({"code": code_input})

                st.subheader("❓Problem Statement")
                st.success(result.problemStatement)

                st.subheader("💻 Language")
                st.success(result.language)

                st.subheader("⚙ Functionality")
                st.write(result.functionality)

                st.subheader("📊 Complexity Level")
                st.metric("Complexity", result.complexity_level)

                st.subheader("📝 Summary")
                st.write(result.summary)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please paste some code first.")
