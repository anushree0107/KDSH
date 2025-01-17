from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
import sys
import time
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from dotenv import load_dotenv
from Retriever.cvpr_retriever import CVPRRulebook
from Retriever.emnlp_retriever import EMNLPRulebook
from Retriever.neurips_retriever import NeurIPSRulebook
from Retriever.kdd_retriever import KDDRulebook
from Retriever.tmlr_retriever import TMLRRulebook

set_llm_cache(InMemoryCache())
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

class GradePaperReview(BaseModel):
    """Binary score for paper acceptance relevance based on review."""
    binary_score: str = Field(
        description="Paper review indicates acceptance into the conference, 'yes' or 'no'"
    )

class PaperGrader:
    def __init__(self, model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY")):
        self.llm = ChatGroq(model=model, api_key=api_key)

        self.structured_llm_grader = self.llm.with_structured_output(GradePaperReview)
        self.system_prompt = """You are a grader tasked with assessing whether the retrieved contexts (rules of the specific conference) are sufficient to provide a proper review for the given section of a research paper. Your role is to analyze the retrieved contexts and determine their adequacy in evaluating the specified section.

            ### Decision-Making Framework:
            1. **Understand the Retrieved Contexts**: Analyze the provided contexts retrieved from the relevant conferences. These contexts include the rules and evaluation criteria specific to each conference. Pay attention to the following:
            - Relevance of the contexts to the paper's focus area.
            - Coverage of necessary criteria, including novelty, technical soundness, clarity, relevance, and validation.
            - Any unique or specific requirements outlined by the conference.

            2. **Analyze the Section of the Paper**:
            - For the specified section (e.g., abstract, methodology, results), assess whether the retrieved contexts provide adequate information to evaluate it effectively.
            - Does the context include clear criteria relevant to this section of the paper?
            - Are there any gaps or missing details in the retrieved contexts that hinder a proper review?

            3. **Compare with User Input**:
            - Evaluate the user's input (paper section details and review comments) alongside the retrieved contexts.
            - Identify any discrepancies or alignments between the contexts and the user-provided section details.

            4. **Make a Judgment**:
            - If the retrieved contexts are sufficient to provide a proper review for the given section, respond 'yes'.
            - If the contexts lack critical information, are vague, or fail to address the requirements for reviewing the section, respond 'no'.

            ### Output Requirements:
            Provide a binary score ('yes' or 'no') based on your analysis. If 'no', briefly explain why the retrieved contexts are insufficient."""


        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", 
                 """Paper Details: \n\n{paper_details}\n\nReview Comments: \n\n{review}"""
                ),
            ]
        )
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader

    def grade_review(self, paper_details, retrieved_contexts):
        
        score = self.retrieval_grader.invoke(
            {"paper_details": paper_details, "review": retrieved_contexts}
        )
        return {"acceptance" : score.binary_score, "review" : retrieved_contexts}

if __name__ == "__main__":
    paper_details = """Methodology : The engineered system includes a camera and a projector connected to a computer on a support. At
each computer round, the system captures an image of the painting and analyzes it to extract the
canvas strokes. This pre-processing is made robust to changes in lighting, ensuring that the interaction
can be used seamlessly in any studio. These strokes then feed into a neural sketcher, which produces
new strokes to be added to the painting. Post-processing is used to project those additions back onto
the canvas.
The neural sketcher is a recurrent neural network, based on a recent improvement to the seminal work
of previous research. It is trained using a sequence of points and a channel encoding for stroke breaks.
The sketcher produces a similar series, which is then converted back into strokes on the original
.
painting. The network was trained using the QuickDraw data set, enabling it to create human-like
strokes. For integration with Tina and Charly’s style, the learning was refined using a sketch database
from previous paintings by the artists."""

    retrieved_contexts = [
        {'text': "However, the lack of hyperparameter optimization and embedding analysis slightly\ndetracts from its rigor.\n● Excitement (4/5):\nThe findings challenge traditional approaches and offer a fresh perspective on\nknowledge injection. The simplicity of the method and its potential impact on\nresearch directions make it exciting.\n● Reproducibility (4/5):\nThe methodology is clear and reproducible, but slight variations may arise due to\nsample variance or reliance on prior hyperparameter settings.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully analyzed the claims and findings and is confident about\nthe paper's strengths and limitations.\nReasons for Acceptance\n1. Novel Insight:\n", 'score': 0.7314755320549011}, 
    ]

    grader = PaperGrader()
    response = grader.grade_review(paper_details=paper_details, retrieved_contexts=retrieved_contexts)
    print(f"Grade for paper acceptance: {response['acceptance']}")
    print("Retrived contexts" , response['review'])
