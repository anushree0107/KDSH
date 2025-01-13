from pathway.xpacks.llm.vector_store import VectorStoreClient
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool


# Define input schema for the tool
class QueryInput(BaseModel):
    query: str = Field(description="The query to search for in the vector store.")


# Define the retriever function
def query_rulebook(query: str) -> str:
    """Queries the Neurips rulebook and retrieves relevant information."""
    output_file = "/home/anushree/KDSH/Retriever/Rulebook/neurips_rule.txt"
    PATHWAY_PORT = 8080  # Ensure the server is running on this port

    client = VectorStoreClient(host="127.0.0.1", port=PATHWAY_PORT)

    # Perform the query
    query_results = client(query)
    return query_results


neurips_retriever_tool = StructuredTool.from_function(
    func=query_rulebook,
    name="NeuripsRetriever",
    description="Useful for retrieving relevant information from the Neurips rulebook based on a query.",
    args_schema=QueryInput,
    return_direct=True
)

# Example usage
print(neurips_retriever_tool.name)
print(neurips_retriever_tool.description)
print(neurips_retriever_tool.args)

if __name__ == "__main__":
    query = "What are the strengths of the accepted paper on ScaleGMN?"
    result = neurips_retriever_tool({"query": query})
    print("Query Results:")
    print(result)

