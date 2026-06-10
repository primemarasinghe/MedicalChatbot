from flask import Flask, render_template, jsonify, request
from flask.cli import load_dotenv
from langchain_groq import ChatGroq 
from src.helper import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain. chains. combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import * 
import os
load_dotenv()

app = Flask(__name__)


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embedding = download_embeddings()
index_name = "medicalbot"
docsearch = PineconeVectorStore.from_existing_index(
    embedding=embedding,
    index_name=index_name)

retriever = docsearch.as_retriever(search_type="similarity",search_kwargs={"k": 3})
chat_model1 = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key= os.getenv("GROQ_API_KEY"),
    temperature=0
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

question_answering_chain = create_stuff_documents_chain(chat_model1, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print("User Input:", input)
    response = rag_chain.invoke({"input": input})
    print("MediBot Response:", response["answer"])
    return str(response["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)