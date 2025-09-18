from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel,Field
from typing import Literal
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence,RunnableParallel,RunnableBranch,RunnableLambda,RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()


pdf='Real-Time_Fire_Surveillance_and_SMS_Alert_System_w_250425_112145 (1).pdf'
loader=PyPDFLoader(pdf)
docs=loader.load()
fullText = "\n".join([d.page_content for d in docs])

# print(fullText)

splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

chunks=splitter.create_documents([fullText])



# embeddings

# embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store=Chroma.from_documents(
        documents=chunks,
    embedding=embeddings,
    collection_name="collection_pdf",
    persist_directory="chroma_db"
)


retriever=vector_store.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 10}
)

llm=ChatGoogleGenerativeAI(model='gemini-2.0-flash-lite',
                             google_api_key=os.getenv("GEMINI_API_KEY"))


prompt=PromptTemplate(
    template=
    """
you are a helpful assistant
Answer ONLY from the provided PDF context.
If the context is insufficient, just say tou don't know.
{context}
Question: {question}
""",
input_variables=['context','question']
)


def format_docs(retrieved_doc):
    context_text="\n\n".join([d.page_content for d in retrieved_doc])
    return context_text


parallelChain=RunnableParallel(
    {
    'context':retriever|RunnableLambda(format_docs),
    'question':RunnablePassthrough()
})
parser=StrOutputParser()
mainChain=parallelChain|prompt|llm|parser
question="Summarize the PDF"
answer=mainChain.invoke(question)

print(answer)

