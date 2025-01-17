import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from Retriever.emnlp_retriever import EMNLPRulebook

from dotenv import load_dotenv
load_dotenv()
from Router_Agent.main_agent import router_agent_response
from Grader.grader_agent import PaperGrader
from Hallucinator.hallucination_agent import HallucinationGrader
from Question_Rewriter.question_rewriter import QueryRewriter
from ReactAgent.react_agent import PaperReviewAgent
from Context_summary import summarization_agent
from ReflectionAgent.reflectionAgent import IntrospectiveAgentManager
from langchain.globals import set_llm_cache
from langchain.cache import InMemoryCache
from MathAgent.math_agent import MathAgent
from langchain_groq import ChatGroq
set_llm_cache(InMemoryCache())


query_list = [
    "Tell me about this paper, should it be accepted in KDD", 
    "Give the rules for getting accepted in TMLR"
]

retriver = EMNLPRulebook()
retrived_contexts = []
for elem in query_list:
    retrived_context = retriver.query_vector_store(elem)
    retrived_contexts.append(retrived_context)

def pipeline(query, retrieved_context):

    possible_conferences = router_agent_response(query)
    print(possible_conferences)
    grader = PaperGrader()
    response = grader.grade_review(paper_details=query, retrieved_contexts=retrieved_context)
    print(f"Grade for paper acceptance: {response['acceptance']}")

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
        
        combined_query = f"{comprehensive_query}\n{query}"
        try:
            answer = react_agent.analyze_paper(combined_query, retrieved_context)
        except Exception as e:
            answer = react_agent.analyze_paper(combined_query, retrieved_context)

        hallucination_agent = HallucinationGrader()
        hallucination_response = hallucination_agent.grade_hallucinations(retrieved_context, answer['output'])

        final_response = None

        if hallucination_response == 'yes':
            agent_manager = IntrospectiveAgentManager(llm_model="llama3-70b-8192", embed_model_name="BAAI/bge-small-en-v1.5")
            introspective_agent = agent_manager.create_introspective_agent(verbose=True)
            combined_context = combined_query
            combined_context += "\n".join([ctx['text'] for ctx in retrieved_context])
            try:
                response = introspective_agent.chat(combined_context)
            except Exception as e:
                response = introspective_agent.chat(combined_context)
            final_response = response['output']
            print("Final Pipeline Response: ", final_response)
        else:
            Mathagent = MathAgent()
            llm = ChatGroq(
                model="llama3-70b-8192", 
                max_retries=5,
                api_key = os.getenv("GROQ_API_KEY")
            )
            checker_response = Mathagent.check_is_math_heavy(query)
            if checker_response.content == '1':
                analysis = Mathagent.analyze_methodology(query)
                sys_prompt = """   
                You are an expert in joining the response of the mathematical analysis and overall analysis.
""" 
                pipeline_response = answer['output']
                joined_query = f"""
                    {pipeline_response} + "\n" + {analysis}
                """
                messages = [
                    ("system", sys_prompt),
                    ("human", joined_query)
                ]
                answer = llm.invoke(messages)
                pipeline_response = answer.content
                print("Final Pipeline response: ", pipeline_response)
            else:
                pipeline_response = answer['output']
                print("Final Pipeline response: ", pipeline_response)
    else:
        query_transformer = QueryRewriter()
        modified_query = query_transformer.apply_hyde(query)
        new_retrieved_context = retriver.query_vector_store(modified_query)
        pipeline(modified_query, new_retrieved_context)


if __name__ == "__main__":
    for i in range(len(query_list)):
        pipeline(query_list[i], retrived_contexts[i])
    