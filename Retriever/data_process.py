import pathway as pw
# from utils.custom_splitter import ContextualMetadataSplitter
# from utils.embedding_client import EmbeddingClient


class DataProcessor:
    def __init__(self):
        self.embedder = EmbeddingClient(cache_strategy=pw.udfs.DefaultCache())
        self.splitter = ContextualMetadataSplitter(chunk_overlap=400, chunk_size=4000)
        # self.splitter = ContextualSplitter(chunk_overlap=400, chunk_size=4000)
        # self.splitter = SummarySplitter(
        #     chunk_size=1024, chunk_overlap=100, llm_provider="openai"
        # )
