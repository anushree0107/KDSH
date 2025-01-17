import os
import ast
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools import TavilySearchResults
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
set_llm_cache(InMemoryCache())

load_dotenv()

tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)


def router_agent_response(query):

    # websearch_response.append(elem["content"])
    prompt = ChatPromptTemplate(
        [
            ("system", """You are a Router Agent for research conferences. Your role is to evaluate the content of a research paper and recommend exactly 2 most relevant conferences from the following list:

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
"""),
            ("human", "{paper_content}\n")
        ]
    )

    input_query = prompt.format(paper_content=query)

    llm = ChatGroq(model="llama3-70b-8192", api_key=os.getenv("GROQ_API_KEY"))
    answer = llm.invoke(input_query)
    response = answer.content
    return response

if __name__ == "__main__":
    text = "This paper introduces a novel algorithm for graph-based fraud detection in large datasets."
    answer = router_agent_response(text)
    print(answer)