from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import ServerlessSpec
from pinecone import Pinecone
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

extracted_data=load_pdf_files(data='data/')
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks=text_splitter(filter_data)

embedding = download_embeddings()

pinecone_api_key = PINECONE_API_KEY
pc = Pinecone(api_key=pinecone_api_key)

index_name = "medicalbot"

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  # Dimension of the embedding vectors
        metric="cosine",  # Similarity metric (e.g., "cosine", "euclidean")
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Serverless configuration
    )
index = pc.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embedding,
    index_name=index_name
)

#load exisiting details from pinecone
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name
)