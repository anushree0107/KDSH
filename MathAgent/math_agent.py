import os
from typing import Optional
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain_groq import ChatGroq
from langchain.cache import InMemoryCache
from langchain.globals import set_llm_cache
import re

set_llm_cache(InMemoryCache())
load_dotenv()

class MathAgent:
    def __init__(self, model_name: str = "mixtral-8x7b-32768"):
        self.llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.0
        )
        self.checker_llm = ChatGroq(
            model_name="llama3-70b-8192",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.01
        )
        self.math_llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model_name=model_name,
            temperature=0.0
        )
        self.tools = load_tools(
            ["llm-math"],
            llm=self.math_llm,
        )
       
        self.agent_chain = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True,
        )

    def check_is_math_heavy(self, content : str):
        system_prompt = """
            You are an expert in judging if the given input text is math heavy or not.
            Final answer will be only 0 or 1.
            if math heavy then 1 else 0.
"""
        messages = [
            ("system", system_prompt),
            ("human", content)
        ]
        response = self.checker_llm.invoke(messages)
        return response


    def extract_methodology(self, paper_text: str) -> Optional[str]:
        """
        Extract methodology section from paper text using regex patterns
        """
        # Common patterns for methodology sections
        patterns = [
            r"(?i)methodology\s*:(.+?)(?=\n\n|\Z)",
            r"(?i)method\s*:(.+?)(?=\n\n|\Z)",
            r"(?i)approach\s*:(.+?)(?=\n\n|\Z)"
        ]
       
        for pattern in patterns:
            match = re.search(pattern, paper_text, re.DOTALL)
            if match:
                return match.group(1).strip()
       
        return None

    def analyze_methodology(self, methodology_text: str) -> str:
        """
        Analyze the methodology section using the math agent
        """
        prompt = f"""Analyze the following methodology section and identify any mathematical or technical components that need verification:

{methodology_text}

Focus on:
1. Numerical calculations or parameters
2. Technical specifications
3. Mathematical models or algorithms
4. Statistical analysis methods

Provide a detailed analysis with mathematical verification where needed.
"""
       
        try:
            response = self.agent_chain.run(prompt)
            return response
        except Exception as e:
            return f"Error analyzing methodology: {str(e)}"

def main():
    # Example usage
    paper_text = """Methodology: The engineered system includes a camera and a projector connected to a computer on a support. At
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

    math_agent = MathAgent()

    checker_response = math_agent.check_is_math_heavy(paper_text)

    print(checker_response.content)
   
    if checker_response.content == '1':
        analysis = math_agent.analyze_methodology(paper_text)
        print("Methodology Analysis:")
        print(analysis)
    else:
        print("No seperate math agent required!!")
    
        

if __name__ == "__main__":
    main()