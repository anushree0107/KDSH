import os
import sys
import re
from typing import List, Optional
from dotenv import load_dotenv
from llama_index.llms.groq import Groq
from llama_index.core.callbacks import CallbackManager
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgent
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
import fitz  # PyMuPDF

class NoveltyCheckAgent:
    def __init__(self, 
                 model_name: str = "llama3-8b-8192",
                 temperature: float = 0.0,
                 embedding_model_name: str = "BAAI/bge-small-en-v1.5",
                 verbose: bool = True):
        """
        Initialize the NoveltyCheckAgent with specified configurations.
        """
        self._setup_environment()
        self.verbose = verbose
        self.model_name = model_name
        self.temperature = temperature
        self.embedding_model_name = embedding_model_name
        self.tools = []
        self.agent = None
        
        self._setup_models()
        self._setup_tools()

    def _setup_environment(self) -> None:
        """Setup environment variables and paths"""
        load_dotenv()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        sys.path.append(parent_dir)

    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        self.llm = Groq(
            model=self.model_name,
            temperature=self.temperature,
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.llm = self.llm

        self.embedding_model = HuggingFaceEmbedding(
            model_name=self.embedding_model_name
        )
        Settings.embed_model = self.embedding_model

        self.callback_manager = CallbackManager([])

    def _setup_tools(self) -> None:
        tavily_tool = TavilyToolSpec(
            api_key=os.getenv("TAVILY_API_KEY"),
        )
        self.tools = tavily_tool.to_tool_list()

    def create_agent(self) -> FunctionCallingAgent:
        """Create and configure the function calling agent"""
        self.agent = FunctionCallingAgent.from_tools(
            tools=self.tools,
            llm=Settings.llm,
            verbose=self.verbose
        )
        return self.agent

    def query_web(self, text: str) -> str:
        """
        Query the web for novelty checking
        
        Args:
            text (str): The text to query
            
        Returns:
            str: The agent's response
        """
        if not self.agent:
            self.create_agent()
        
        try:
            response = self.agent.chat(text)
            return str(response)
        except Exception as e:
            return f"Error during query: {str(e)}"

    def extract_sections_from_pdf(self, pdf_path: str) -> dict:
        """
        Extract specific sections from a PDF
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            dict: Extracted sections
        """
        sections = {
            "Methodology": "",
            "Background": "",
            "Algorithm": "",
            "Experiments": ""
        }
        
        with fitz.open(pdf_path) as pdf:
            for page in pdf:
                text = page.get_text()
                for section in sections.keys():
                    pattern = rf"{section}:.*?(?=\n[A-Z])"
                    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
                    if match:
                        sections[section] += match.group(0)
        
        return sections

    def process_papers(self, folder_path: str) -> None:
        """
        Process all PDFs in a folder and check for novelty
        
        Args:
            folder_path (str): Path to the folder containing PDF files
        """
        pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
        for pdf_file in pdf_files:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"\nProcessing: {pdf_file}")
            
            sections = self.extract_sections_from_pdf(pdf_path)
            for section, content in sections.items():
                if content.strip():
                    print(f"\nSection: {section}\n{content[:500]}...")
                    response = self.query_web(content[:500])  # Limit query to 500 chars
                    print(f"\nWeb Search Result for {section}:\n{response}")

def main():
    agent = NoveltyCheckAgent(verbose=True)
    papers_folder = "/home/anushree/KDSH/Papers"
    
    if os.path.exists(papers_folder):
        agent.process_papers(papers_folder)
    else:
        print(f"Folder '{papers_folder}' not found!")

if __name__ == "__main__":
    main()
