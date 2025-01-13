#  Pipeline Components

This pipeline is designed to deliver accurate, contextually enriched responses by leveraging a series of specialized agents. Below is an overview of its key components:

## Key Features

### 1. **Router Agent**
   - Routes queries to the appropriate source:
     - **Vectorstore:** For retrieving internal documents.
     - **Web Search Agent:** For fetching real-time data from the web.

### 2. **Web Search Agent**
   - Utilizes Tavily search to retrieve up-to-date information from the internet.

### 3. **Retrieval Agent**
   - Enhances data retrieval accuracy.

### 4. **Grader Agent**
   - Validates the relevance and quality of the retrieved documents.

### 5. **Question Rewriter Agent**
   - Refines ambiguous or unclear queries for better processing by downstream agents.

### 6. **Generator Agent**
   - Integrates LangChain, Tavily Search, and ChatGroq to execute ReAct-based reasoning and acting for answering complex queries.

### 8. **Hallucination Grader Agent**
   - Detects and minimizes hallucinated or incorrect information in responses.

### 9. **Introspective Agent**
   - Reviews responses for:
     - Enhanced accuracy.
     - Reduced toxicity and improved clarity.

### 10. **Joiner Agent**
   - Combines generated responses to assign papers to suitable journals based on selection criteria.

---

This modular pipeline ensures high precision and context-aware answers, making it an efficient and reliable system for journal selection.
