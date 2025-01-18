import os
import hashlib
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor, load_tools
from langchain_community.tools import TavilySearchResults
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain_community.cache import GPTCache

class RouterAgent:
    def __init__(self, llm_model="llama3-70b-8192", cache_dir="cache_data"):
        load_dotenv()
        self.llm_model = llm_model
        self.cache_dir = cache_dir
        self._initialize_cache()
        self.system_prompt = self._create_system_prompt()
        self.prompt_template = self._create_prompt_template()
        self.agent_executor = self._initialize_agent_executor()

    def _initialize_cache(self):
        """Initializes the cache system for the LLM."""
        set_llm_cache(InMemoryCache())

        def get_hashed_name(name):
            return hashlib.sha256(name.encode()).hexdigest()

        def init_gptcache(cache_obj: Cache, llm: str):
            hashed_llm = get_hashed_name(llm)
            cache_obj.init(
                pre_embedding_func=get_prompt,
                data_manager=manager_factory(manager="map", data_dir=f"{self.cache_dir}_{hashed_llm}"),
            )

        set_llm_cache(GPTCache(init_gptcache))

    def _create_system_prompt(self):
        """Defines the system prompt for the Router Agent."""
        return """
        You are a Router Agent for research conferences. Your role is to evaluate the content of a research paper and recommend exactly 2 most relevant conferences from the following list:

        1. CVPR - Computer Vision and Pattern Recognition: Focuses on computer vision, image processing, and related tasks.
        2. EMNLP - Empirical Methods in Natural Language Processing: Specializes in NLP methods, language models, and linguistics-based research.
        3. NeurIPS - Neural Information Processing Systems: Covers a broad range of machine learning, AI, and computational neuroscience topics.
        4. KDD - Knowledge Discovery and Data Mining: Focuses on data mining, large-scale data analysis, and applications of AI in knowledge discovery.
        5. TMLR - Transactions on Machine Learning Research: Emphasizes general advancements and methods in machine learning.

        When analyzing the paper, think step-by-step about the content, methodology, and domain of the research. Match the paper's focus areas to the scope of the conferences. 

        **Important**: Output only the Python list of exactly 2 conferences. Do not include any analysis or explanation.

        ### Few-shot Examples:

        **Example 1**
        Paper Content: "The paper discusses a novel method for improving transformer models in NLP tasks."
        **Output**: ["EMNLP", "NeurIPS"]

        **Example 2**
        Paper Content: "The paper proposes a new framework for unsupervised image segmentation."
        **Output**: ["CVPR", "NeurIPS"]

        **Example 3**
        Paper Content: "This paper introduces a novel algorithm for graph-based fraud detection in large datasets."
        **Output**: ["KDD", "NeurIPS"]

        **Example 4**
        Paper Content: "The study presents a novel convolutional neural network for better generalization in visual object recognition tasks."
        **Output**: ["CVPR", "NeurIPS"]

        For each new paper, evaluate its content and output only the Python list with the 2 most relevant conferences, e.g., ["EMNLP", "NeurIPS"].
        You have to give 2 conference names at any cost.
        """

    def _create_prompt_template(self):
        """Creates the chat prompt template."""
        return ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("placeholder", "{agent_scratchpad}"),
                ("human", "{paper_content}"), 
            ]
        )

    def _initialize_agent_executor(self):
        """Initializes the agent executor."""
        llm = ChatGroq(
            model=self.llm_model,
            api_key=os.getenv("GROQ_API_KEY")
        )

        tool = load_tools([
            "arxiv"
        ])

        agent = create_tool_calling_agent(llm=llm, tools=tool, prompt=self.prompt_template)

        return AgentExecutor(
            agent=agent,
            tools=tool,
            verbose=True,
            handle_parsing_errors=True
        )

    def route_paper(self, paper_content):
        """Routes the paper content to the most relevant conferences.

        Args:
            paper_content (str): The content of the research paper.

        Returns:
            list: A Python list of 2 most relevant conferences.
        """
        response = self.agent_executor.invoke({
            "paper_content": paper_content
        })

        return response["output"]

if __name__ == "__main__":
    router = RouterAgent()
    paper_text = "This paper introduces a novel algorithm for graph-based fraud detection in large datasets."
    conferences = router.route_paper(paper_text)
    print(conferences)