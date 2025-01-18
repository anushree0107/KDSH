import os
from typing import List, Dict
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import Settings
from llama_index.agent.coa import CoAAgentWorker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from langchain_community.utilities.arxiv import ArxivAPIWrapper
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
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

# ... existing cache-related imports and setup ...

class PaperReviewAgent:
    def __init__(self):
        load_dotenv()
        self._setup_models()
        self.tools = []
        self.setup_tools()
        self.worker = self.setup_agent()
        
    def _setup_models(self) -> None:
        """Initialize LLM and embedding models"""
        Settings.llm = Groq(
            model="llama3-70b-8192",
            temperature=0.0,
            api_key=os.getenv("GROQ_API_KEY")
        )
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="BAAI/bge-small-en-v1.5"
        )

    def setup_tools(self) -> None:
        """Setup tools for the agent"""
        arxiv = ArxivAPIWrapper()
        arxiv_tool = QueryEngineTool(
            query_engine=arxiv,
            metadata=ToolMetadata(
                name="arxiv_tool",
                description="Tool for searching and analyzing academic papers from arXiv. Use this for finding and evaluating research papers.",
            ),
        )
        self.tools.append(arxiv_tool)

    def setup_agent(self) -> CoAAgentWorker:
        """Setup the CoA agent with the tools"""
        system_prompt = """
        You are an expert reviewer and evaluator for academic research papers submitted to leading journals and conferences. 
        Your task is to assess the quality of a paper based on its novelty, strengths, weaknesses, and relevance to the target journal. 
        When given a retrieved context about a paper, analyze it to explain why the paper might get selected by the journal. 
        
        Ensure your explanation highlights:
        1. The key strengths of the paper (novelty, impactful findings, methodological advancements)
        2. How the paper aligns with the journal's focus and audience
        3. Any limitations acknowledged and their impact on quality
        4. The potential impact or contribution to the field
        5. Specific reasons why the paper might not meet journal criteria
        
        Tailor your response to match the specific journal's criteria and expectations.
        """
        
        return CoAAgentWorker.from_tools(
            tools=self.tools,
            llm=Settings.llm,
            system_prompt=system_prompt,
            verbose=True
        )

    def analyze_paper(self, query: str, retrieved_contexts: List[Dict[str, str]]) -> str:
        """Analyze a paper using the CoA agent"""
        combined_context = "\n\n".join([ctx['text'] for ctx in retrieved_contexts])
        
        # Combine the query and context for the agent
        full_query = f"Based on the following context:\n{combined_context}\n\nAnalyze: {query}"
        
        agent = self.worker.as_agent()
        response = agent.chat(full_query)
        
        # Add error handling for response
        if response is None or str(response).strip() == '':
            return "Error: Received empty response from agent"
        
        try:
            # If response is already a string, return it directly
            if isinstance(response, str):
                return response
            # If response is a dict or other object, convert to string
            return str(response)
        except Exception as e:
            return f"Error processing response: {str(e)}"

# ... existing main block remains the same ...
if __name__ == "__main__":
    retrieved_contexts = [
        {'text': "However, the lack of hyperparameter optimization and embedding analysis slightly\ndetracts from its rigor.\n● Excitement (4/5):\nThe findings challenge traditional approaches and offer a fresh perspective on\nknowledge injection. The simplicity of the method and its potential impact on\nresearch directions make it exciting.\n● Reproducibility (4/5):\nThe methodology is clear and reproducible, but slight variations may arise due to\nsample variance or reliance on prior hyperparameter settings.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully analyzed the claims and findings and is confident about\nthe paper's strengths and limitations.\nReasons for Acceptance\n1. Novel Insight:\n", 'score': 0.7314755320549011}, 
        {'text': 'evaluation details are underspecified, which could introduce challenges in\nreproducing the results exactly.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully evaluated the important aspects of the paper and is\nconfident in the assessment.\nReasons for Acceptance\nThis paper makes a significant contribution to the field by addressing the challenges of\nNLP for low-resource languages like Finnish. The creation of LLMs, the extension of\nmultilingual models like BLOOM, and the development of Fin-Bench demonstrate a\ncomprehensive and impactful effort. The practical evaluations, along with the\nopen-source release of scripts and data, enhance its value to the community. These factors,\n', 'score': 0.7564890384674072}, 
        {'text': 'In addition to creating standalone Finnish models, they extend the BLOOM model to\nFinnish while maintaining English performance, demonstrating effective multilingual\nadaptation.\n3. Holistic Evaluation:\nThe paper goes beyond task-level evaluation by testing for biases, human\nalignment, and toxicity in the models, offering practical insights for real-world\napplications and cautioning their use in production systems.\n4. Benchmark Creation:\nThe introduction of Fin-Bench provides a valuable resource for evaluating Finnish\nLLMs, contributing to the broader NLP community working on low-resource\nlanguages.\n5. Detailed Methodology:\nThe authors provide comprehensive details about the training process, including\nhyperparameters, architecture, and hardware, ensuring that others can replicate or\n', 'score': 0.7659016251564026}
    ]

    methodology_query = """Analyze this methodology for NeurIPS acceptance:
    
    Methodology: The engineered system includes a camera and a projector connected to a computer on a support. At
    each computer round, the system captures an image of the painting and analyzes it to extract the
    canvas strokes. This pre-processing is made robust to changes in lighting, ensuring that the interaction
    can be used seamlessly in any studio. These strokes then feed into a neural sketcher, which produces
    new strokes to be added to the painting. Post-processing is used to project those additions back onto
    the canvas.
    The neural sketcher is a recurrent neural network, based on a recent improvement to the seminal work
    of previous research. It is trained using a sequence of points and a channel encoding for stroke breaks.
    The sketcher produces a similar series, which is then converted back into strokes on the original
    painting. The network was trained using the QuickDraw data set, enabling it to create human-like
    strokes. For integration with Tina and Charly's style, the learning was refined using a sketch database
    from previous paintings by the artists."""

    comprehensive_query = """Analyze this paper for NeurIPS acceptance:

    STRENGTHS:
    1. Key novelty and technical contributions
    2. Quality of methodology and experiments
    3. Impact potential and significance

    LIMITATIONS:
    1. Technical gaps or weaknesses
    2. Areas needing improvement
    3. Presentation issues (if any)

    VERDICT:
    Would you recommend acceptance? Why/why not?

    Support your points with brief examples from the paper."""

    combined_query = f"{methodology_query}\n\n{comprehensive_query}"

    agent = PaperReviewAgent()

    

    result = agent.analyze_paper(combined_query, retrieved_contexts)

    print(result["output"])