# import os
# from llama_index.retrievers.pathway import PathwayRetriever
# from llama_index.core import Settings
# from llama_index.llms.groq import Groq
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
# from dotenv import load_dotenv

# load_dotenv()

# embedding_model = HuggingFaceEmbedding(model_name="BAAI/bge-reranker-v2-m3")
# Settings.embed_model = embedding_model
# Settings.llm = Groq(model="llama3-8b-8192", api_key=os.getenv("GROQ_API_KEY"))

# retriever = PathwayRetriever(
#     url="https://python.langchain.com/api_reference/community/vectorstores/langchain_community.vectorstores.pathway.PathwayVectorClient.html"
# )


# from llama_index.core.query_engine import RetrieverQueryEngine

# query_engine = RetrieverQueryEngine.from_args(
#     retriever,
# )
# response = query_engine.query("Tell me about Pathway")
# print(str(response))

import pathway as pw
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import PathwayVectorClient

data_sources = []
data_sources.append(
    pw.io.gdrive.read(object_id="1dMwMM4JgYLkIyg1FRnMDaYrtiIbu-nQa", service_user_credentials_file="/home/anushree/KDSH/data-connector/credentials.json", with_metadata=True))

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
embeddings_model = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
vector_server = PathwayVectorServer(
    *data_sources,
    embedder=embeddings_model,
    splitter=text_splitter,
)
vector_server.run_server(host="127.0.0.1", port="8765", threaded=True, with_cache=False)
client = PathwayVectorClient(
    host="127.0.0.1",
    port="8765",
)
query = "What is Pathway?"
docs = client.similarity_search(query)