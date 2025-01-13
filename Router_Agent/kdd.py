import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""
    sections = {
        "title": "",
        "abstract": "",
        "introduction": "",
        "related work": "",
        "dataset": "",
        "methodology": "",
        "method": "",
        "experimental results": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "references": ""
    }
    
    # Extract title from first page
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    
    # Look for title in the first few blocks
    for block in blocks[:3]:  # Usually title is in first few blocks
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Check if text is bold (flags & 2^4 indicates bold text)
                    if span["flags"] & 16 and not any(word in span["text"].lower() for word in ["abstract", "introduction"]):
                        sections["title"] += span["text"] + " "
    
    sections["title"] = sections["title"].strip()
    print(f"Extracted title: {sections['title']}")
    
    # Get all text from the PDF with better formatting preservation
    for page in doc:
        text_content += page.get_text("text", sort=True) + "\n"
    
    # Split text into lines while preserving formatting
    lines = text_content.split('\n')
    current_section = None
    section_content = ""
    
    # Common section header patterns
    section_patterns = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "1. introduction", "i. introduction", "1 Introduction"],
        "related work": ["related work", "background", "2. related work", "ii. related work"],
        "dataset": ["dataset", "data", "2. dataset", "ii. dataset"],
        "methodology": ["methodology", "method", "proposed method", "3. methodology", "iii. methodology", "approach"],
        "experimental results": ["experimental results", "experiments", "4. experiments", "iv. experiments"],
        "results": ["results", "4. results", "iv. results"],
        "discussion": ["discussion", "5. discussion", "v. discussion"],
        "conclusion": ["conclusion", "conclusions", "6. conclusion", "vi. conclusion"],
        "references": ["references", "bibliography"]
    }
    
    # Process each line with better section detection
    for i, line in enumerate(lines):
        line = line.strip()
        line_lower = line.lower()
        
        # Check if this line is a section header
        is_header = False
        next_section = None
        
        for section, patterns in section_patterns.items():
            if any(pattern == line_lower or pattern == line_lower.strip('0123456789. ') for pattern in patterns):
                if current_section:
                    # Save the content of the previous section
                    sections[current_section] += section_content.strip() + "\n"
                current_section = section
                section_content = ""
                is_header = True
                break
            
            # Look ahead for next section to determine section boundaries
            if i < len(lines) - 1:
                next_line_lower = lines[i + 1].lower().strip()
                if any(pattern == next_line_lower or pattern == next_line_lower.strip('0123456789. ') for pattern in patterns):
                    next_section = section
        
        # Add content to current section if we're in one and it's not a header
        if current_section and not is_header:
            if line.strip():  # Only add non-empty lines
                section_content += line + " "
        
        # If we've found the next section header, save current section content
        if next_section and current_section:
            sections[current_section] += section_content.strip() + "\n"
            section_content = ""
    
    # Add the last section's content
    if current_section and section_content:
        sections[current_section] += section_content.strip()
    
    # Clean up the sections
    for section in sections:
        if section != "title":  # Don't process title section
            # Remove section headers from content
            for pattern in section_patterns.get(section, []):
                sections[section] = sections[section].replace(pattern, "")
            # Clean up extra whitespace while preserving paragraphs
            sections[section] = ' '.join(sections[section].split())
    
    return sections

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

    def route_title_abstract(self, title, abstract, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Research Paper Title: {title}\n"
            f"Abstract: {abstract}\n"
            "Analyze the title and abstract, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result
    
    def route_introduction(self, introduction, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Introduction: {introduction}\n"
            "Analyze the introduction, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result
    
    def route_dataset(self, dataset, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Dataset: {dataset}\n"
            "Analyze the dataset, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result
    
    def route_methodology(self, methodology, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Methodology: {methodology}\n"
            "Analyze the methodology, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result
    
    def route_experimental_results(self, experimental_results, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Experimental Results: {experimental_results}\n"
            "Analyze the experimental results, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result
    
    def route_conclusion(self, conclusion, journal_name):
        question = (
            f"Journal: {journal_name}\n"
            f"Conclusion: {conclusion}\n"
            "Analyze the conclusion, and provide a recommendation whether this paper fits the journal."
        )
        result = self.question_router.invoke({"question": question})
        return result

# Path to the dataset folder
dataset_folder = "/mnt/c/Users/HP/OneDrive/Desktop/kdsh-task-2/KDSH/Papers"

# Check if the dataset folder exists
if not os.path.exists(dataset_folder):
    raise FileNotFoundError(f"Dataset folder not found at {dataset_folder}")

# Function to load research papers from the dataset folder
def load_research_papers(folder_path):
    papers = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            paper = extract_text_from_pdf(file_path)
            papers[file_name] = paper
    return papers

# Function to determine if a paper fits into a journal based on its title and abstract
def check_paper_fit_with_router_agent(router, paper, journal_name):
    title = paper["title"]
    abstract = paper["abstract"]
    introduction = paper["introduction"]
    dataset = paper["dataset"]
    methodology = paper["methodology"]
    experimental_results = paper["experimental results"]
    conclusion = paper["conclusion"]

    result_title_abstract = router.route_title_abstract(title, abstract, journal_name)
    print(f"Result for abstract of '{title}' in journal '{journal_name}': {result_title_abstract}")

    result_introduction = router.route_introduction(introduction, journal_name)
    print(f"Result for introduction of '{title}' in journal '{journal_name}': {result_introduction}")

    result_dataset = router.route_dataset(dataset, journal_name)
    print(f"Result for dataset of '{title}' in journal '{journal_name}': {result_dataset}")

    result_methodology = router.route_methodology(methodology, journal_name)
    print(f"Result for methodology of '{title}' in journal '{journal_name}': {result_methodology}")

    result_experimental_results = router.route_experimental_results(experimental_results, journal_name)
    print(f"Result for experimental results of '{title}' in journal '{journal_name}': {result_experimental_results}")

    result_conclusion = router.route_conclusion(conclusion, journal_name)
    print(f"Result for conclusion of '{title}' in journal '{journal_name}': {result_conclusion}")

# Load papers
papers = load_research_papers(dataset_folder)

# Initialize the Router
router = Router(api_key=groq_api_key)

for key in papers.keys():
    journal_name = "KDD"
    check_paper_fit_with_router_agent(router, papers[key], journal_name)