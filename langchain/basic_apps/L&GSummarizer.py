from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


from pydantic import BaseModel, Field

class LGWriteup(BaseModel):
    summary: str = Field(description="Summary of the chapter in around 150 words")
    positive_ref: str = Field(description="Positive life reference related to the chapter in 150-200 words")
    negative_ref: str = Field(description="Negative life reference related to the chapter in 150-200 words")
    commitment: str = Field(description="Commitment/action plan in 80-100 words")


parser = PydanticOutputParser(pydantic_object=LGWriteup)

template = PromptTemplate(
    template="""
You are an assistant helping a student prepare their weekly L&G session write-up. 
Write about 600 words in 4 parts: summary, positive reference, negative reference, and commitment.

Chapter: {chapter}

{format_instructions}
""",
    input_variables=["chapter"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.8
)

chain = template | model | parser

st.title("📖 L&G Write-up Generator")
st.write("Generate your 600-word weekly reflection with summary, personal references, and commitment.")

chapter_text = st.text_area("Paste Chapter Summary or Key Points:", height=200)

if st.button("Generate Write-up"):
    if chapter_text.strip():
        with st.spinner("Generating write-up..."):
            result = chain.invoke({"chapter": chapter_text})
        st.subheader("✅ Your Write-up")
        st.markdown(f"**Summary**\n\n{result.summary}")
        st.markdown(f"**Positive Reference**\n\n{result.positive_ref}")
        st.markdown(f"**Negative Reference**\n\n{result.negative_ref}")
        st.markdown(f"**Commitment**\n\n{result.commitment}")
    else:
        st.warning("Please paste some chapter content first.")