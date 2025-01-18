import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from Retriever.emnlp_retriever import EMNLPRulebook
from final_pipeline import pipeline
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Define RouteQuery class
class RouteQuery(BaseModel):
    """Route a research paper to the most relevant academic journal."""

    conference: Literal["CVPR", "EMNLP", "NeurIPS", "KDD", "TMLR"] = Field(
        ..., description="The conference to which the paper is most relevant."
    )

    rationale: str = Field(
        ..., description="Provide a short rationale for the conference recommendation."
    )

class SelectorAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.system_prompt = """You are an expert at selecting the most relevant conference for a research paper.\nUse the two possible conferences, a relevant retrieved context, and the response from our AI pipeline, given for each section of the paper, to select the most relevant conference for the entire paper, and provide a rationale of around 100 words for the same."""
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )
        self.question_router = self.route_prompt | self.structured_llm_router

    def select_conference(self, router_journals, retrieved_context, pipeline_response):
        question = (
            f"Possible conferences: {', '.join(router_journals)}\n"
            f"Relevant context: {', '.join(retrieved_context)}\n"
            f"Pipeline response: {', '.join(pipeline_response)}\n"
        )
        result = self.question_router.invoke({"question": question})
        return result
