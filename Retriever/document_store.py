import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging

import pathway as pw
import requests
from config import IndexerConfig as Config
from pathway.stdlib.indexing.bm25 import TantivyBM25Factory
from pathway.stdlib.indexing.hybrid_index import HybridIndexFactory
from pathway.stdlib.indexing.nearest_neighbors import UsearchKnnFactory
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.xpacks.llm.servers import DocumentStoreServer
from pydantic import InstanceOf
# from utils.data_processor import DataProcessor
# from utils.metadata_helper import MetadataFilterGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)



class DocumentStoreServerWrapper:
    def __init__(self):

        config = Config()

        self.host = "127.0.0.1"
        self.port = 8765
        self.server = None
        self.client = None
        self.with_cache: bool = True

        self.cache_backend: InstanceOf[pw.persistence.Backend] = (
            pw.persistence.Backend.filesystem(".cache/embedding-cache")
        )
        self.terminate_on_error: bool = False

    def create_server(
        self,
        data,
        embedder=None,
        splitter=None,
    ):
        default_processor = DataProcessor()
        if embedder is None:
            embedder = default_processor.embedder
        if splitter is None:
            splitter = default_processor.splitter

        try:
            self.usearch_knn_factory = UsearchKnnFactory(
                dimensions=embedder.get_embedding_dimension(),
                reserved_space=1000,
                connectivity=0,
                expansion_add=0,
                expansion_search=0,
                embedder=embedder,
            )
        except Exception as e:
            logger.error(f"Error creating UsearchKnnFactory: {str(e)}")
            raise

        try:
            self.bm25_factory = TantivyBM25Factory()
        except Exception as e:
            logger.error(f"Error creating BM25Factory: {str(e)}")
            raise

        self.retriever_factories = [self.bm25_factory, self.usearch_knn_factory]

        try:
            self.hybrid_index_factory = HybridIndexFactory(
                retriever_factories=self.retriever_factories
            )
            self.document_store = DocumentStore(
                *data,
                retriever_factory=self.hybrid_index_factory,
                splitter=splitter,
            )
        except Exception as e:
            logger.error(f"Error creating DocumentStore Instance: {str(e)}")
            raise

        try:
            self.server = DocumentStoreServer(
                host=self.host, port=self.port, document_store=self.document_store
            )
            logger.info("Server created successfully")
        except Exception as e:
            logger.error(f"Error creating server: {str(e)}")
            raise

    def run_server(self, with_cache=True, threaded=False):
        try:
            print(f"Starting server on {self.host}:{self.port}")
            self.server.run(
                with_cache=with_cache,
                threaded=threaded,
                cache_backend=self.cache_backend,
                terminate_on_error=self.terminate_on_error,
            )
            print("Server started successfully")
        except Exception as e:
            logger.error(f"Error running server: {str(e)}")
            raise


class DocumentStoreClientWrapper:
    def __init__(self):
        config = Config()
        self._create_client()
        self.vector_db_url = config.VECTOR_DB_URL
        self.retriever_url = f"{self.vector_db_url}/v1/retrieve"
        self.metadata_filter_generator = MetadataFilterGenerator()

    def _create_client(self):
        logger.info("Client created successfully")

    def get_docs(self, payload):
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                self.retriever_url, data=json.dumps(payload), headers=headers
            )

            if response.status_code == 200:
                results = response.json()
                logger.info(f"Query Results:")
                logger.info(json.dumps(results, indent=2))
                return results
            else:
                logger.error(
                    f"Failed to retrieve results. Status code: {str(response.status_code)}"
                )
                logger.error("Response:", response.text)
                return None
        except Exception as e:
            logger.error(f"Error querying document store: {str(e)}")
            return None