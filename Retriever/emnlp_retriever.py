import time
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from sentence_transformers import SentenceTransformer
from pathway.xpacks.llm import embedders

class EMNLPRulebook:
    def __init__(self):
        self.data_sources = []
        self.vector_server = None
        self.client = None

        # Automatically save the rulebook to the file
        self.read_data_sources()
        self.initialize_vector_server()
        self.initialize_client()

    def read_data_sources(self):
        """Reads data from the output file into data_sources."""
        self.data_sources.append(
            pw.io.fs.read(
                "/mnt/c/Users/HP/OneDrive/Desktop/kdsh-task-2/KDSH/Retriever/Rulebook/emnlp_rulebook.txt",
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
            min_tokens=50,  # Ensure small meaningful chunks
            max_tokens=150,  # Cover larger reviews in one chunk
            encoding_name="cl100k_base",  # Token encoding for compatibility 
        )
        # Define custom embedder class
        class SentenceTransformerEmbedder:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name, trust_remote_code=True)

            def __call__(self, texts):
                return self.model.encode(texts)

        # Replace embedder initialization
        embedder = embedders.SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        
        PATHWAY_PORT = 8765

        self.vector_server = VectorStoreServer(
            *self.data_sources,
            embedder=embedder,
            splitter=text_splitter,
        )
        self.vector_server.run_server(
            host="127.0.0.1", 
            port=PATHWAY_PORT, 
            threaded=True, 
            with_cache=True,
        )
        time.sleep(60)  # Wait for server initialization

    def initialize_client(self):
        """Initializes the VectorStoreClient."""
        self.client = VectorStoreClient(
            host="127.0.0.1",
            port=8765,
            timeout=60
        )
    
    def query_vector_store(self, query):
        """Queries the vector store and returns results."""
        query_results = self.client(query)
        return query_results