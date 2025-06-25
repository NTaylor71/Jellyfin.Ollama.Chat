from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Qdrant
from langchain_ollama import Ollama
import os

llm = Ollama(
    model=os.getenv("OLLAMA_MODEL", "mistral"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
)

vectorstore = Qdrant(
    url=os.getenv("VECTORDB_URL", "http://vectordb:6333"),
    collection_name=os.getenv("VECTORDB_COLLECTION", "jellyfin_rag"),
    prefer_grpc=False
)

retriever = vectorstore.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

query = "What movies were released in 1999?"
result = qa.run(query)
print("🔎 Query:", query)
print("🧠 Answer:", result)
