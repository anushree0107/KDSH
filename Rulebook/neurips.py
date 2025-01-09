import logging
import sys
import time

logging.basicConfig(stream=sys.stderr, level=logging.WARN, force=True)

import os
from dotenv import load_dotenv

load_dotenv()

import pathway as pw

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "./sample_documents",
        format="binary",
        mode="streaming",
        with_metadata=True,
    )
)

from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer

from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class HuggingFaceEmbedder:
    def __init__(self, model_name="nvidia/embedding-v2"):
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def __call__(self, text):
        # Handle input being a single string or a list of strings
        if isinstance(text, str):
            text = [text]

        # Tokenize input
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        # Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use the mean of the last hidden state as the embedding
            embeddings = outputs.last_hidden_state.mean(dim=1).numpy()

        # Return embeddings as a list of dictionaries
        return [{"embedding": np.array(embedding, dtype=np.float32)} for embedding in embeddings]

# Instantiate the Hugging Face Embedder
embedder = HuggingFaceEmbedder()

PATHWAY_PORT = 8765

text_splitter = TokenCountSplitter()
embedder = HuggingFaceEmbedder()

vector_server = VectorStoreServer(
    *data_sources,
    embedder=embedder,
    splitter=text_splitter,
)

vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=True, with_cache=False)
time.sleep(30)

client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

# Save the rulebook content
rulebook_content = """  

1. Novelty
Accepted: The paper must present new algorithms, applications, or theories that address problems that have not been extensively explored or provide significant advancements in the field.
Rejected: The paper fails to introduce novel concepts or repeats well-established approaches without substantial innovation. If the proposed work has been adequately covered in existing literature, it should highlight its unique contributions or provide detailed comparative analyses to demonstrate its novelty.
Weak Points:

Lack of differentiation from prior work, such as using previously established methods without innovation.
The contribution seems incremental and does not break new ground.
Similar methods or concepts have been explored in past studies without any additional insights.
2. Technical Contribution
Accepted: The paper should provide a robust theoretical analysis or proofs for the proposed methods, offering high-quality empirical results on real-world or complex problems. The approach must show clear advantages over existing methods.
Rejected: If the proposed method lacks rigorous theoretical validation, empirical experiments, or does not address real-world problems, it may be rejected. Superficial novelty or unsubstantiated claims reduce the technical soundness of the paper.
Weak Points:

The paper's methodology lacks clear validation or is based on underdeveloped experimental design.
Insufficient empirical data or reliance on incomplete benchmarks.
The experiments do not provide concrete evidence of the proposed method’s superiority or real-world applicability.
3. Insight and Impact
Accepted: The paper must demonstrate meaningful insights into algorithms, applications, or problems in the relevant domain. It should highlight its broader impact, contributing to advancing the field of machine learning or related areas.
Rejected: If the paper lacks insight into the problem and does not significantly contribute to advancing the field, it will be rejected. Clear articulation of key takeaways for the scientific community is essential.
Weak Points:

The work lacks a clear explanation of how it advances the state-of-the-art.
No clear impact on the broader scientific community or application domains.
Contributions are not easily generalizable or fail to push the boundaries of current research.
4. Interdisciplinary Appeal
Accepted: For conferences like NeurIPS, the work must bridge artificial and natural systems where appropriate, appealing to a broad audience across different domains.
Rejected: A paper is rejected if it lacks cross-disciplinary appeal or fails to connect with broader research areas, limiting its relevance or interest.
Weak Points:

Overly specialized work with limited appeal beyond a niche audience.
No consideration of broader interdisciplinary applications or connections to other fields.
5. Evaluation Metrics
Accepted: Use strong baselines and relevant datasets for comparison. The chosen evaluation metrics must be well justified, and the paper should explain discrepancies in results.
Rejected: The paper is rejected if it lacks proper baselines, uses inadequate or irrelevant datasets, or if the evaluation metrics are poorly justified or inconsistently applied.
Weak Points:

The use of weak or non-representative baselines.
Inconsistent or insufficient evaluation on widely accepted datasets.
Lack of explanation for the choice of metrics or performance anomalies.
6. Application Relevance
Accepted: The paper must clearly demonstrate how the proposed method is relevant for real-world applications and highlight its practical benefits, challenges, and limitations.
Rejected: If the paper fails to address real-world applicability, lacks practical insights, or does not discuss limitations, it risks rejection.
Weak Points:

Overemphasis on theoretical models without clear real-world relevance.
Failure to address how the method could be used in practice, or ignoring potential limitations.
Lack of real-world case studies or experiments on diverse applications.
7. Clarity and Reproducibility
Accepted: The paper must be clearly written, accessible to a broad audience, and provide sufficient detail for others to reproduce the results. The motivation, experiment design, and takeaways must be explicitly outlined.
Rejected: Papers with unclear explanations, incomplete or missing details, or inadequate support for reproducing results will be rejected.
Weak Points:

Ambiguous explanations or complex jargon without sufficient clarification.
Missing or incomplete code, data, or descriptions preventing reproducibility.
Unclear structure or lacking coherence in the paper’s presentation.
8. Literature Review and Related Work
Accepted: A thorough and well-organized review of related work is necessary. The paper must address connections and differences between existing literature and the proposed method, showcasing how it contributes to the field.
Rejected: Failure to engage with existing literature or superficial treatment of related work leads to rejection. The work must be contextualized within the broader body of research.
Weak Points:

Sparse or superficial related work sections that do not critically engage with existing studies.
Missing key references or failing to explain how the paper differs from previous work.
9. Ethical Considerations
Accepted: Papers must explicitly address any potential ethical concerns, including societal impact and possible misuse of the proposed method. Both positive and negative implications should be considered.
Rejected: Papers that overlook ethical considerations or fail to address potential harms will be rejected.
Weak Points:

Lack of discussion on ethical implications or societal impact.
Ignoring the potential risks and misuse of the proposed method.
10. Soundness and Presentation
Accepted: The paper must be technically solid with no major flaws in logic, methodology, or experiment design. Figures, tables, and visualizations should support key points and maintain high readability standards.
Rejected: Major flaws in the methodology, poor presentation, or excessive errors in the paper lead to rejection.
Weak Points:

Logical or methodological flaws that undermine the results.
Low-quality figures or poorly designed tables that confuse rather than clarify.
Typos, unclear terminology, or weak formatting affecting readability.
Final Decision Criteria: Acceptance or Rejection
Accept: Papers are accepted if they meet the above criteria in a balanced and rigorous manner. A clear contribution to the field, strong empirical evidence, and real-world relevance are crucial for acceptance.
Reject: Papers are rejected if they lack novelty, technical depth, or clarity. Lack of solid experimentation, inadequate literature review, and failure to justify methodological choices may also lead to rejection. Papers with serious flaws or limited contributions to the field should be rejected.
"""

output_file = "neurips_rulebook.txt"
with open(output_file, "w") as f:
    f.write(rulebook_content)

print("\n--- NeurIPS Rulebook ---\n")
print(rulebook_content)
print(f"\nGuidelines saved to {output_file}")

query = "What is Deep Learning? "
query_results = client(query)

print("\n--- Retrieved Documents ---\n")
print(query_results)

pw.run()