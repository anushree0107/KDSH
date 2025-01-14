from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )    

class HallucinationGrader:
    def __init__(self):
        load_dotenv()
        self.llm = ChatGroq(
            model="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY")
        )
        self.structured_llm_grader = self.llm.with_structured_output(GradeHallucinations)
        
        self.system_prompt = """You are a grader assessing whether an LLM's paper review is grounded in the provided context about the paper. 
        
        Give a binary score 'yes' or 'no':
        - 'Yes' means the review is supported by the retrieved context
        - 'No' means the review contains claims not supported by the context"""
        
        self.hallucination_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("human", """Retrieved Context: {documents}

Paper Review Generated: {generation}

Is this review grounded in the retrieved context? Answer only with 'yes' or 'no'."""),
            ]
        )
        self.hallucination_grader = self.hallucination_prompt | self.structured_llm_grader
        
    def grade_hallucinations(self, documents, generation):
        """
        Grade whether the paper review contains hallucinations.
        
        Args:
            documents (list): List of retrieved context dictionaries
            generation (str): The generated paper review to evaluate
            
        Returns:
            str: 'yes' or 'no' indicating if the review is grounded in facts
        """
        
        combined_context = "\n\n".join([ctx['text'] for ctx in documents])

        result = self.hallucination_grader.invoke(
            {
                "documents": combined_context,
                "generation": generation
            }
        )
        
        return result.binary_score

if __name__ == "__main__":
    retrieved_contexts = [
        {'text': "However, the lack of hyperparameter optimization and embedding analysis slightly\ndetracts from its rigor.\n● Excitement (4/5):\nThe findings challenge traditional approaches and offer a fresh perspective on\nknowledge injection. The simplicity of the method and its potential impact on\nresearch directions make it exciting.\n● Reproducibility (4/5):\nThe methodology is clear and reproducible, but slight variations may arise due to\nsample variance or reliance on prior hyperparameter settings.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully analyzed the claims and findings and is confident about\nthe paper's strengths and limitations.\nReasons for Acceptance\n1. Novel Insight:\n", 'score': 0.7314755320549011}, 
        {'text': 'evaluation details are underspecified, which could introduce challenges in\nreproducing the results exactly.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully evaluated the important aspects of the paper and is\nconfident in the assessment.\nReasons for Acceptance\nThis paper makes a significant contribution to the field by addressing the challenges of\nNLP for low-resource languages like Finnish. The creation of LLMs, the extension of\nmultilingual models like BLOOM, and the development of Fin-Bench demonstrate a\ncomprehensive and impactful effort. The practical evaluations, along with the\nopen-source release of scripts and data, enhance its value to the community. These factors,\n', 'score': 0.7564890384674072}, 
        {'text': 'In addition to creating standalone Finnish models, they extend the BLOOM model to\nFinnish while maintaining English performance, demonstrating effective multilingual\nadaptation.\n3. Holistic Evaluation:\nThe paper goes beyond task-level evaluation by testing for biases, human\nalignment, and toxicity in the models, offering practical insights for real-world\napplications and cautioning their use in production systems.\n4. Benchmark Creation:\nThe introduction of Fin-Bench provides a valuable resource for evaluating Finnish\nLLMs, contributing to the broader NLP community working on low-resource\nlanguages.\n5. Detailed Methodology:\nThe authors provide comprehensive details about the training process, including\nhyperparameters, architecture, and hardware, ensuring that others can replicate or\n', 'score': 0.7659016251564026}
    ]
    
    example_generation = """Based on the analysis, this paper shows strong potential for NeurIPS acceptance.
    The methodology demonstrates clear novelty in combining computer vision with neural sketching,
    and the results are reproducible with proper documentation of hyperparameters."""
    
    grader = HallucinationGrader()
    
    print(grader.grade_hallucinations(retrieved_contexts, example_generation)) 