from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()


class NewsSummarizer(BaseModel):
    headline:str=Field(description="Main headline of the article")
    summary:str=Field(description="Concise 3-4 line summary of the article showcasing all the major information")
    category:str=Field(description="category of news like politicals, sports, technology , etc")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Overall sentiment of the article"
    )

parser=PydanticOutputParser(pydantic_object=NewsSummarizer)

template= PromptTemplate(
    template="""You are an independent news auditor and reporter. Your job is to critically analyze, fact-check, and report news in a clear, unbiased, and professional tone. You should highlight the core facts, identify possible bias or misinformation, and provide a balanced summary.
    "Summarize the following news article into structured fields.
Article:{article}
{format_instructions}""",
input_variables=['article'],
partial_variables={'format_instructions':parser.get_format_instructions()}
)



model=ChatGoogleGenerativeAI(model='gemini-2.5-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))


article="""
Apple has officially launched the iPhone 16 series at its annual September event. 
The new lineup features improved cameras, faster processors, and enhanced AI-powered software. 
Analysts expect strong sales, especially in Asian markets, though some criticize the high pricing. 
Shares of Apple rose slightly after the announcement.
"""

model_chain= template|model|parser
# result=model_chain.invoke({"article":article})

# print(result)
st.set_page_config(page_title="📰 News Summarizer", layout="centered")

st.title("📰 AI News Summarizer")
st.write("Paste any news article below and get a structured summary.")

article = st.text_area("Enter News Article", height=250)

if st.button("Summarize"):
    if article.strip():
        with st.spinner("Analyzing article..."):
            try:
                result = model_chain.invoke({"article": article})

                st.subheader("📌 Headline")
                st.success(result.headline)

                st.subheader("📝 Summary")
                st.write(result.summary)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("📂 Category", result.category)
                with col2:
                    st.metric("💡 Sentiment", result.sentiment)

            except Exception as e:
                st.error(f"Error: {str(e)}")
    else:
        st.warning("Please paste a news article first.")