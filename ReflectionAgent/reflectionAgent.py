import os
from dotenv import load_dotenv
from llama_index.agent.introspective import IntrospectiveAgentWorker, SelfReflectionAgentWorker
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.tools.tavily_research.base import TavilyToolSpec
load_dotenv()

class IntrospectiveAgentManager:
    def __init__(self, llm_model: str, embed_model_name: str):
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.llm_model = llm_model
        self.embed_model_name = embed_model_name


    def create_introspective_agent(self, verbose: bool = True):
        tavily_tool = TavilyToolSpec(api_key=self.tavily_api_key)

        llm = Groq(
            model=self.llm_model,
            temperature=0.0,
            api_key=self.groq_api_key
        )

        Settings.llm = llm
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.embed_model_name)

        self_reflection_agent_worker = SelfReflectionAgentWorker.from_defaults(
            llm=llm,
            verbose=verbose
        )

        tool_list = tavily_tool.to_tool_list()
        main_agent_worker = FunctionCallingAgentWorker.from_tools(
            tools=tool_list,
            llm=llm,
            verbose=verbose
        )

        introspective_worker_agent = IntrospectiveAgentWorker.from_defaults(
            reflective_agent_worker=self_reflection_agent_worker,
            main_agent_worker=main_agent_worker,
            verbose=verbose
        )

        return introspective_worker_agent.as_agent(verbose=verbose)

def main():
    llm_model = "llama3-70b-8192"
    embed_model_name = "BAAI/bge-small-en-v1.5"

    agent_manager = IntrospectiveAgentManager(
        llm_model=llm_model,
        embed_model_name=embed_model_name
    )

    retrieved_contexts = [
        {'text': "However, the lack of hyperparameter optimization and embedding analysis slightly\ndetracts from its rigor.\n● Excitement (4/5):\nThe findings challenge traditional approaches and offer a fresh perspective on\nknowledge injection. The simplicity of the method and its potential impact on\nresearch directions make it exciting.\n● Reproducibility (4/5):\nThe methodology is clear and reproducible, but slight variations may arise due to\nsample variance or reliance on prior hyperparameter settings.\n● Ethical Concerns: None identified.\n● Reviewer Confidence (4/5):\nThe reviewer has carefully analyzed the claims and findings and is confident about\nthe paper's strengths and limitations.\nReasons for Acceptance\n1. Novel Insight:\n", 'score': 0.7314755320549011}, 
    ]

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

    combined_context = "This is the retrieved context " + "\n" 
    combined_context += "\n".join([ctx['text'] for ctx in retrieved_contexts])

    combined_context += comprehensive_query 

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

    combined_context += paper_details

    introspective_agent = agent_manager.create_introspective_agent(verbose=True)

    response = introspective_agent.chat(combined_context)
    
    response = str(response)

    print("Final Response : ", response)
if __name__ == "__main__":
    main()