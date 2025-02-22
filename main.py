import os
import bs4
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GOOGLE_API_KEY bulunamadı! Lütfen çevre değişkenlerini kontrol edin.")

chat = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=api_key, temperature=0.4)

loader = WebBaseLoader(
    web_path="https://en.wikipedia.org/wiki/Quantum_mechanics",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(class_="mw-content-container"))
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
splits = text_splitter.split_documents(docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_store = Chroma.from_documents(documents=splits, embedding=embedding_model)
retriever = vector_store.as_retriever()

# Prompt Template
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    You are an AI assistant. Use the following retrieved context to answer the user's question.
    If the context does not contain the answer, say "I don’t know."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
)

def format_retrieved_docs(docs):
    return "\n\n".join(doc.page_content if hasattr(doc, "page_content") else str(doc) for doc in docs)

# RAG Pipeline
chain = (
    {"context": retriever | format_retrieved_docs, "question": RunnablePassthrough()}
    | prompt
    | chat
    | StrOutputParser()
)

# starting FastAPI 
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)

class QueryRequest(BaseModel):
    question: str

@app.post("/rag")
async def rag_api(request: QueryRequest):
    result = chain.invoke(request.question)
    return {"answer": result}
