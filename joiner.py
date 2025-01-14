from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
import os

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Define RouteQuery class
class RouteQuery(BaseModel):
    """Route a research paper to the most relevant academic journal."""

    review: str = Field(
        ..., description="Provide a short review of the paper."
    )

class JoinerAgent:
    def __init__(self):
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        self.system_prompt = """You are an expert at reviewing research papers.\nUse the overviews of the different sections of the paper like abstract, introduction, methodology etc to generate a summary and a review of the paper with its strengths, weaknesses, suggestions and scores."""
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )
        self.question_router = self.route_prompt | self.structured_llm_router

    def join_sections(self, sections):
        question = (
            f"Title: {sections['title']}\n"
            f"Abstract: {sections['abstract']}\n"
            f"Introduction: {sections['introduction']}\n"
            f"Methodology: {sections['methodology']}\n"
            f"Results: {sections['results']}\n"
            f"Conclusion: {sections['conclusion']}\n"
        )
        result = self.question_router.invoke({"question": question})
        return result
