# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv
load_dotenv()
from sklearn.metrics.pairwise import cosine_similarity as cs

# embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = SentenceTransformer('all-MiniLM-L6-v2')


documents = [
    "Sachin Tendulkar – Known as the 'God of Cricket', he is the highest run-scorer in international cricket.",
    "Virat Kohli – A modern batting great, admired for his aggressive style and unmatched consistency.",
    "M.S. Dhoni – The 'Captain Cool' who led India to World Cup victories in both T20 (2007) and ODI (2011).",
    "AB de Villiers – Nicknamed 'Mr. 360' for his ability to play shots all around the ground."
]


query="Tell me about MS Dhoni"

# doc_embedding=embeddings.embed_documents(documents)
# query_embedding=embeddings.embed_query(query)
doc_embedding=model.encode(documents)
query_embedding=model.encode([query]).reshape(1, -1)


res=cs(query_embedding,doc_embedding)[0]
print(res)