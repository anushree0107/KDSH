import os
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

class PaperReviewAgent:
    def __init__(self):
        load_dotenv()

        self.tools = [TavilySearchResults(max_results=1)]

        self.system_prompt = """
        You are an expert reviewer and evaluator for academic research papers submitted to leading journals and conferences. Your task is to assess the quality of a paper based on its novelty, strengths, weaknesses, and relevance to the target journal. When given a retrieved context about a paper, you must analyze it to explain why the paper might get selected by the journal. Ensure your explanation highlights:

        1. The key strengths of the paper (e.g., novelty, impactful findings, methodological advancements).
        2. How the paper aligns with the journal's focus and audience.
        3. Any limitations acknowledged in the paper and their impact on its overall quality.
        4. The potential impact or contribution of the paper to the field.
        5. Specific reasons why the paper might not meet the journal's criteria, such as methodological flaws, limited impact, or other weaknesses.


        When evaluating a query about a specific journal (e.g., NeurIPS), tailor your response to match the criteria and expectations of that journal. Your response must be clear, concise, and reflective of the retrieved context. Avoid unnecessary details and provide precise reasoning for the paper's selection.
        """

        self.llm = ChatGroq(
            model="llama3-70b-8192",
            max_retries=5,
            api_key=os.getenv("GROQ_API_KEY")
        )

        self.prompt = hub.pull("hwchase17/react")

        self.agent = create_react_agent(llm=self.llm, tools=self.tools, prompt=self.prompt)

        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)

    def analyze_paper(self, query, retrieved_contexts):
        combined_context = "\n\n".join([ctx['text'] for ctx in retrieved_contexts])

        answer = self.agent_executor.invoke({"input": query, "context": combined_context})

        return answer

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