import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from llama_index.core import PromptTemplate
from llama_index.llms.groq import Groq
from llama_index.core import Settings

import sys
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from dotenv import load_dotenv
from Grader.grader_agent import PaperGrader
import hashlib
from gptcache import Cache
from langchain.globals import set_llm_cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()


def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )


set_llm_cache(GPTCache(init_gptcache))

load_dotenv()

class HyDETransformer:
    def __init__(self, model_name="llama3-70b-8192"):
        self.llm = Groq(
            model=model_name, 
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.llm = self.llm
        self.hyde = HyDEQueryTransform(include_original=True)

    def generate_hypothetical_answer(self, query: str):
        """
        Generate a hypothetical document using HyDE transformation
        """
        query_bundle = self.hyde.run(query)
        return query_bundle.custom_embedding_strs[0]

class QueryRewriter:
    def __init__(self):
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.hyde_transformer = HyDETransformer()
        self.grader = PaperGrader()
        
        self.system_prompt = """You are a question re-writer that converts an input question to a better version that is optimized 
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        
        self.question_rewriter = self.rewrite_prompt | self.llm | StrOutputParser()

    def rewrite_query(self, query):
        """
        First attempt basic query rewriting
        """
        better_query = self.question_rewriter.invoke({"question": query})
        return better_query

    def apply_hyde(self, query):
        """
        Apply HyDE transformation to generate a hypothetical document
        """
        hyde_result = self.hyde_transformer.generate_hypothetical_answer(query)
        enhanced_query = f"{query}\n{hyde_result}"
        return enhanced_query

    # def process_query(self, query, router_response, max_attempts=3):

    #     current_query = query
    #     attempt = 0
        
    #     while attempt < max_attempts:
    #         # Grade the current query
    #         grader_response = self.grader.grade_review(current_query, router_response)
            
    #         if grader_response["acceptance"].lower() == "yes":
    #             return current_query
                
    #         current_query = self.apply_hyde(current_query)
    #         attempt += 1
            
    #     return current_query

if __name__ == "__main__":
    rewriter = QueryRewriter()
    
    original_query = """Methodology: The engineered system includes a camera and a projector connected to a computer on a support. 
    The neural sketcher is a recurrent neural network, based on a recent improvement to the seminal work of previous research."""
    
    
    final_query = rewriter.apply_hyde(original_query)
    print("Final Enhanced Query:", final_query)