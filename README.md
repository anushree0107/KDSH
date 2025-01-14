# KDSH 2025: Automating Research Paper Evaluation and Conference Selection

## Introduction

Team Data Hackers introduces **ARCS (Agentic RAG for Conference Selection)**, a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed to automate the evaluation of research papers and recommend suitable conferences such as EMNLP, CVPR, and NeurIPS. By leveraging advanced natural language processing techniques and a multi-agent system, ARCS ensures accurate, context-rich, and efficient recommendations.

## Features

1. **Automated Paper Evaluation:** Assesses research papers based on clarity, rigor, and relevance.
2. **Conference Recommendation:** Suggests appropriate conferences for paper submissions.
3. **Multi-Agent System:** Incorporates specialized agents for tasks like query routing, retrieval, grading, and refinement.
4. **Advanced Embedding and Retrieval:** Utilizes state-of-the-art models for semantic similarity and efficient data handling.
5. **Dynamic Query Processing:** Employs chain-of-thought prompting and prompt caching for enhanced response generation.

## Pipeline Components

ARCS employs a modular architecture with the following key components:

- **Router Agent:** Directs queries to the appropriate processing path, determining whether to use internal retrieval or web search.
- **Retrieval Agent:** Fetches relevant documents using specialized retrievers tailored to specific conferences.
- **Grader Agent:** Evaluates the relevance and quality of retrieved documents concerning the query.
- **Question Rewriter Agent:** Refines unclear or ambiguous queries to enhance retrieval effectiveness.
- **Hallucination Grader Agent:** Assesses the factual accuracy of generated responses to mitigate hallucinations.
- **Introspective Agent:** Performs comprehensive analysis to ensure the response meets user needs.

![Pipeline Diagram](https://imgur.com/a/70xHgBh)

---

## Getting Started

1. **Clone the repository**:

   ```bash
   git clone https://github.com/anushree0107/KDSH.git
   cd KDSH
   ```

2. **Create a Virtual Environment** (optional but recommended):

   ```bash
   python -m venv env
   source env/bin/activate   # On Windows: env\Scripts\activate
   ```

3. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Keys**:

   To run the pipeline, create a .env file in the root directory(it is already provided, if api exhausts then one needs to provide his own) and add the required API keys in the following format:
   ```bash
   GROQ_API_KEY=<your_groq_api_key>
   TAVILY_API_KEY=<your_tavily_api_key>
   ```

5. **Run the application**:

   ```bash
   python main_recommendation.py
   ```
   
