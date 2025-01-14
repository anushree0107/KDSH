import os
import sys
from dotenv import load_dotenv
load_dotenv()
from Router_Agent.main_agent import router_agent_response
from Grader.grader_agent import PaperGrader
from Hallucinator.hallucination_agent import HallucinationGrader
from Question_Rewriter.question_rewriter import QueryRewriter
from ReactAgent.react_agent import PaperReviewAgent
from ReflectionAgent.reflectionAgent import IntrospectiveAgentManager


def pipeline(text):

    print("given query: ", text)
    possible_conferences = router_agent_response(text)
    retrieved_contexts = []
    grader = PaperGrader()
    retrieved_contexts.append({"text": "This is a context from the rulebook"})
    response = grader.grade_review(paper_details=text, retrieved_contexts=retrieved_contexts)
    print(f"Grade for paper acceptance: {response['acceptance']}")
    retrieved_contexts = response['review']

    if response['acceptance'] == 'yes':
        react_agent = PaperReviewAgent()
        comprehensive_query = f"""Analyze this paper for acceptance in these following conferences:
    {possible_conferences}
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
        
        combined_query = f"{comprehensive_query}\n{text}"
        try:
            answer = react_agent.analyze_paper(combined_query, retrieved_contexts)
        except Exception as e:
            answer = react_agent.analyze_paper(combined_query, retrieved_contexts)

        hallucination_agent = HallucinationGrader()
        hallucination_response = hallucination_agent.grade_hallucinations(retrieved_contexts, answer['output'])

        if hallucination_response == 'yes':
            agent_manager = IntrospectiveAgentManager(llm_model="llama3-70b-8192", embed_model_name="BAAI/bge-small-en-v1.5")
            introspective_agent = agent_manager.create_introspective_agent(verbose=True)
            combined_context = combined_query
            combined_context += "\n".join([ctx['text'] for ctx in retrieved_contexts])
            try:
                response = introspective_agent.chat(combined_context)
            except Exception as e:
                response = introspective_agent.chat(combined_context)
            print("final response: ", response['output'])
        else:
            print("final response: ", answer['output'])
    else:
        query_transformer = QueryRewriter()
        modified_query = query_transformer.apply_hyde(text)
        return pipeline(modified_query)
    

if __name__ == "__main__":
    example_query = """
This document outlines our contribution to the ActivityNet Challenge, focusing on
active speaker detection. We employ a 3D convolutional neural network (CNN)
for feature extraction, combined with an ensemble of temporal convolution and
LSTM classifiers to determine whether a person who is visible is also speaking.
The results demonstrate substantial improvements compared to the established
baseline on the AVA-ActiveSpeaker dataset.
"""
    pipeline(example_query)
