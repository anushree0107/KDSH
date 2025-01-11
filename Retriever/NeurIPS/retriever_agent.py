import time
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
from sentence_transformers import SentenceTransformer
from pathway.xpacks.llm import embedders

class NeuripsRulebook:
    def __init__(self, rulebook_content, output_file_path):
        self.rulebook_content = rulebook_content
        self.output_file_path = output_file_path
        self.data_sources = []
        self.vector_server = None
        self.client = None

        # Automatically save the rulebook to the file
        self.save_rulebook_to_file()
        self.read_data_sources()
        self.initialize_vector_server()
        self.initialize_client()

    def save_rulebook_to_file(self):
        """Saves the rulebook content to a specified file."""
        with open(self.output_file_path, "w", encoding="utf-8") as f:
            f.write(self.rulebook_content)
        
        print(f"\nGuidelines saved to {self.output_file_path}")

    def read_data_sources(self):
        """Reads data from the output file into data_sources."""
        self.data_sources.append(
            pw.io.fs.read(
                self.output_file_path,
                format="binary",
                mode="streaming",
                with_metadata=True
            )
        )
        print("Data Connector Initialized")
        pw.run()

    def initialize_vector_server(self):
        """Initializes the VectorStoreServer with data sources and an embedder."""
        text_splitter = TokenCountSplitter()

        # Define custom embedder class
        class SentenceTransformerEmbedder:
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
                self.model = SentenceTransformer(model_name, trust_remote_code=True)

            def __call__(self, texts):
                # Convert embeddings to numpy array and ensure correct type
                import numpy as np
                embeddings = self.model.encode(texts)
                return np.array(embeddings, dtype=np.float32)

        # Replace embedder initialization
        embedder = embedders.SentenceTransformerEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        
        PATHWAY_PORT = 8765

        self.vector_server = VectorStoreServer(
            *self.data_sources,
            embedder=embedder,
            splitter=text_splitter,
        )
        self.vector_server.run_server(
            host="127.0.0.1", 
            port=PATHWAY_PORT, 
            threaded=True, 
            with_cache=False,
        )
        time.sleep(60)  # Wait for server initialization

    def initialize_client(self):
        """Initializes the VectorStoreClient."""
        self.client = VectorStoreClient(
            host="127.0.0.1",
            port=8765
        )
    
    def query_vector_store(self, query):
        """Queries the vector store and returns results."""
        query_results = self.client(query)
        return query_results

# Example Usage in another file:
def get_query_results(query):
    # Initialize the NeuripsRulebook instance (this step happens automatically)
    rulebook_content = """  
    Rejected Paper Review (Federated Learning):
    ● Summary: The paper explores asynchronous federated contextual bandit and reinforcement learning, proposing algorithms that use exploration-based bonus functions. Finite-time convergence and communication complexities are analyzed.
    ● Strengths:
    ○ Introduces trigger-based communication for multi-agent systems.
    ○ Focuses on asynchronous sampling and communication with a server.
    ● Weaknesses:
    ○ Key notations and concepts (e.g., in Theorems 4.3 and 5.1) are unclear.
    ○ Poor explanation of the bonus term computation oracle.
    ○ Confusing statements and multiple typos (e.g., Line 141, Line 154).
    ● Rating: 5 (Borderline Accept).
    Accepted Paper Review (Stable Diffusion Optimization):
    ● Summary: Proposes a three-stage post-training optimization (Refiner, Retriever, Composer) to personalize stable diffusion models for prompts, achieving enhanced image generation quality.
    ● Strengths:
    ○ Novel approach focusing on model adaptation instead of prompt engineering.
    ○ Well-designed optimization pipeline with promising experimental results.
    ● Weaknesses:
    ○ Higher resource usage compared to static stable diffusion models.
    ○ Comparison fairness questioned due to Stylus personalization process.
    ● Rating: 7 (Accept).
    Summary of Reviews:
    Accepted Paper Review (ScaleGMN):
    ● Summary: This work introduces ScaleGMNs, GNN-based metanetworks that extend permutation equivariance to account for scaling symmetries in input neural networks' parameters. These networks are expressive enough to simulate forward and backward passes and demonstrate improved performance over prior approaches without using data augmentation or random Fourier features.
    ● Strengths:
    ○ Excellent writing and clear theoretical setup.
    ○ Strong theoretical results supporting scaling equivariances and expressive power.
    ○ Significant empirical improvements, particularly in INR classification, without reliance on unfair optimization tricks.
    ○ Interesting findings, such as the bidirectional version's varied performance and robustness without additional features.
    ● Weaknesses:
    ○ Limited scope of experimental tasks, lacking tests on equivariant tasks like INR editing.
    ○ Some inconsistencies in describing ScaleInv equations and canonicalization approaches.
    ● Rating: 8 (Strong Accept).
    ● Confidence: 4 (High confidence).
    Accepted Paper Review (Causal Discovery in Ancestral Graphs):
    ● Summary: The paper proposes a greedy search-and-score algorithm for causal structure discovery in ancestral graphs, leveraging a decomposition of the likelihood function into multivariate cross-information over ac-connected components. It demonstrates scalability and experimental efficacy on synthetic and benchmark datasets.
    ● Strengths:
    ○ Introduces a scalable, practical algorithm for causal structure discovery.
    ○ Shows higher precision compared to existing algorithms on experimental datasets.
    ● Weaknesses:
    ○ Theoretical guarantees for algorithm convergence are missing.
    ○ The main theorem lacks novelty as it overlaps with prior work.
    ○ Sensitivity to noise in cross-information computations is a concern.
    ○ The connection between theoretical results and algorithm implementation needs better elucidation.
    ● Rating: 5 (Borderline Accept).
    ● Confidence: 3 (Moderate confidence).
    Accepted Paper Review:
    1. Strengths:
    ○ Introduces a novel and practical framework for quantifying and optimizing Chain-of-Thought (CoT) reasoning.
    ○ Extensive experiments validate generalizability and utility.
    ○ Provides actionable optimization strategies for CoT.
    2. Weaknesses:
    ○ Limited to four tasks; broader evaluation is needed for generalization.
    ○ Relies on task difficulty as input, which might not always be readily available.
    3. Rating and Assessment:
    ○ The paper is technically solid and impactful within its area.
    ○ Clear presentation and well-justified claims.
    ○ Review highlights specific strengths and offers constructive feedback for improvement.
    ○ High confidence in the assessment (rating: 7/10, accept).
    Rejected Paper Review:
    1. Strengths:
    ○ Reviewer acknowledges the potential contributions but struggles with understanding the methodology and presentation.
    ○ Transparent admission of difficulty in following the paper.
    2. Weaknesses:
    ○ Lacks clarity in explanation, especially for key terms, concepts, and contributions.
    ○ Insufficient details on experimental procedures and methodology.
    ○ Over-reliance on jargon without clear definitions.
    ○ No adequate comparison to related work, particularly Yang et al.’s 2024 study.
    ○ Numerous unaddressed questions about methodology, datasets, and contributions.
    3. Rating and Assessment:
    ○ The paper's potential is overshadowed by its lack of clarity and organization.
    ○ Reviewer shows limited confidence due to unclear methodology (rating: 4/10, borderline reject).
    ○ Suggestions focus heavily on improving readability and detailing methodology.
    Accepted Paper : Self-Preference and Recognition in LLMs
    ● Key Idea: Investigates whether language models exhibit self-preference due to self-recognition.
    ● Strengths:
    ○ Insightful separation of self-recognition and self-preference.
    ○ Thoughtful evaluation, including human cross-checking and diverse control tasks.
    ○ Addressed potential confounds and ordering effects.
    ● Weaknesses:
    ○ Limited task scope (summarization); potential dataset memorization confound.
    ○ Lack of results for larger models like Llama 70B; scaling trends unclear.
    ● Suggestions:
    ○ Explore other domains to reduce confounding.
    ○ Add statistical tests for significance claims.
    ○ Mitigation experiments (e.g., anti-bias prompts) would be valuable.
    ● Rating: 7 (Accept)
    ○ Technically solid with significant impact, strong evaluation, and reproducibility.
    Rejected Paper Review:
    The paper introduces a greedy search-and-score algorithm for causal structure discovery in ancestral graphs, accommodating directed and bidirected edges. It decomposes the likelihood function into multivariate cross-information over ac-connected components, building on prior head-and-tail factorization work. The authors claim this approach provides a novel theoretical foundation for causal graph discovery and demonstrate its application through experimental evaluations.
    Key Strengths
    1. Algorithm Development:
    ○ Novel empirical algorithm motivated by theoretical decomposition of the likelihood function.
    ○ Scalable to graphs with several dozen vertices and links.
    2. Experimental Validation:
    ○ Tested on synthetic datasets and bnlearn benchmarks.
    ○ Demonstrates higher precision and comparable recall to the MIIC algorithm.
    Key Weaknesses
    1. Lack of Theoretical Guarantees:
    ○ No convergence guarantees for the proposed algorithm.
    ○ Algorithm relies heavily on MIIC and lacks independence from its base.
    2. Sensitivity to Noise:
    ○ Multivariate cross-information may be highly sensitive to noise, particularly with large variable sets.
    3. Novelty Concerns:
    ○ Theorem 1 is equivalent to a decomposition from prior work, questioning its originality.
    4. Limited Explanation:
    ○ Insufficient clarity on how Theorem 1 directly informs the proposed algorithm.
    """

    output_file = "/mnt/c/Users/HP/OneDrive/Desktop/kdsh-task-2/KDSH/Retriever/Rulebook/neurips_rulebook.txt"
    
    # Instantiating the class will handle everything automatically
    neurips = NeuripsRulebook(rulebook_content, output_file)
    
    # Query the vector store and return the results
    return neurips.query_vector_store(query)