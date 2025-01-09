import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import pdfplumber  # Library for reading PDFs

# Load environment variables from .env file
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the .env file")

# Define RouteQuery class
class RouteQuery(BaseModel):
    """Route a research paper to the most relevant academic journal."""

    decision: Literal["Fits", "Does not fit"] = Field(
        ..., description="Determine if the paper fits the journal based on its title and abstract."
    )
    explanation: str = Field(
        ..., description="Provide a short explanation for the decision."
    )

# Define Router class
class Router:
    def __init__(self, api_key):
        self.llm = ChatGroq(
            model="mixtral-8x7b-32768",
            temperature=0,  # Set the temperature for deterministic output
            max_tokens=None,  # No limit on the number of tokens
            timeout=None,  # No timeout
            max_retries=2,  # Retry up to 2 times if the request fails
        )
        self.system_prompt = """You are an expert at routing research papers to the most suitable academic journal.\nUse the provided journal name, paper title, and abstract to make an informed decision."""
        self.structured_llm_router = self.llm.with_structured_output(RouteQuery)
        self.route_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", "{question}"),
            ]
        )
        self.question_router = self.route_prompt | self.structured_llm_router

    def route_paper(self, paper_title, paper_abstract, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Research Paper Title: {paper_title}\n"
            f"Abstract: {paper_abstract}\n"
            "Analyze the title and abstract, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result

# Path to the dataset folder
dataset_folder = "/home/anushree/KDSH/Papers"

# Check if the dataset folder exists
if not os.path.exists(dataset_folder):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_folder}")

# Function to load research papers from the dataset folder
def load_research_papers(folder_path):
    papers = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            with pdfplumber.open(file_path) as pdf:
                text = ''.join(page.extract_text() for page in pdf.pages if page.extract_text())
                papers[file_name] = text
    return papers

# Function to determine if a paper fits into a journal based on its title and abstract
def check_paper_fit_with_router_agent(router, paper_title, paper_abstract, journal_name):
    result = router.route_paper(paper_title, paper_abstract, journal_name)
    return result

# Load papers
papers = load_research_papers(dataset_folder)

# Initialize the Router
router = Router(api_key=groq_api_key)

# Example: Using the router agent on the first paper in the folder
for file_name, content in papers.items():
    paper_title = file_name.replace(".pdf", "")  # Use the filename as the title
    paper_abstract = content[:500]  # Use the first 500 characters as the abstract
    journal_name = "EMNLP"
    result = check_paper_fit_with_router_agent(router, paper_title, paper_abstract, journal_name)
    print(f"Result for paper '{paper_title}' in journal '{journal_name}': {result}")