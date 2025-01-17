import time
import os
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from sentence_transformers import SentenceTransformer
from pathway.xpacks.llm import embedders
from typing import List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pathway.xpacks.llm.rerankers import CrossEncoderReranker


load_dotenv()

class EMNLPRulebook:
    def __init__(self):
        self.data_sources = []
        self.vector_server = None
        self.client = None
        self.rerankers = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2")
        self.read_data_sources()
        self.initialize_vector_server()
        self.initialize_client()
    


    def read_data_sources(self):
        """Reads data from the output file into data_sources."""
        self.data_sources.append(
            pw.io.fs.read(
                "/home/anushree/KDSH/Retriever/Rulebook/master_rulebook.txt",
                format="binary",
                mode="streaming",
                with_metadata=True
            )
        )
        print("Data Connector Initialized")
        pw.run()

    def initialize_vector_server(self):
        """Initializes the VectorStoreServer with data sources and an embedder."""
        text_splitter = TokenCountSplitter(
            min_tokens=50,  
            max_tokens=150,  
            encoding_name="cl100k_base",  
        )
       
        class SentenceTransformerEmbedder:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name, trust_remote_code=True)

            def __call__(self, texts):
                return self.model.encode(texts)

       
        embedder = embedders.SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        
        PATHWAY_PORT = 8020

        print(type(self.data_sources[0]))

        self.vector_server = VectorStoreServer(
            *self.data_sources,
            embedder=embedder,
            splitter=text_splitter,
            # doc_post_processors=[self.rerankers]
        )
        self.vector_server.run_server(
            host="127.0.0.1", 
            port=PATHWAY_PORT, 
            threaded=True, 
            with_cache=True,
        )
        time.sleep(60)  

    def initialize_client(self):
        """Initializes the VectorStoreClient."""
        self.client = VectorStoreClient(
            host="127.0.0.1",
            port=8020,
            timeout=60
        )
    def query_vector_store(self, query)->List:
        """Queries the vector store and returns results."""
        query_results = self.client(query)
        final_results = []
        for result in query_results:
            final_results.append(
                {
                    "text": result["text"],
                    "score": result["dist"]
                }
            )
        return final_results
    

    

class EMNLPAgent:
    def __init__(self):
        # self.rulebook = EMNLPRulebook()
        # self.retriever_tool = self.rulebook.setup_tools()
        self.agent = ChatGroq(
            model = "llama3-70b-8192",
            max_retries = 5,
            api_key = os.getenv("GROQ_API_KEY")
        )
    def query(self, query, retrieved_results):
        # retrieved_results = self.rulebook.query_vector_store(query)
        final_prompt = f"""
            This is the query of the user : {query}
            These are the retrieved results from the rulebook:
            {retrieved_results}
        """
        messages = [
            ("system", "Think yourself as a EMNLP reviewer, based on the tool results and your knowledge, what would be the best answer to the following question?"),
            ("user", final_prompt)
        ]
        response = self.agent.invoke(messages)
        return response.content
    
    
if __name__ == "__main__":
    rulebook = EMNLPRulebook()
    queries = [
        "What are the rules to getting selected in EMNLP?",
        "Hello bhai",
        "peace hain"
    ]
    results = []
    for elem in queries:
        res = rulebook.query_vector_store(elem)
        results.append(res)

    print("retrieved results: ", results)
    
    # agent = EMNLPAgent()
    # response = agent.query(query, results)
    # print("response: ", response)