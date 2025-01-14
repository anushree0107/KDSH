import sys
import os
import fitz
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
from dotenv import load_dotenv
load_dotenv()
from Router_Agent.main_agent import router_agent_response
from Grader.grader_agent import PaperGrader
from Hallucinator.hallucination_agent import HallucinationGrader
from Question_Rewriter.question_rewriter import QueryRewriter
from ReactAgent.react_agent import PaperReviewAgent
from ReflectionAgent.reflectionAgent import IntrospectiveAgentManager


def pipeline(text, content, file, key):

    print("given query: ", text)
    possible_conferences = router_agent_response(text)
    retrieved_contexts = []
    grader = PaperGrader()
    retrieved_contexts.append({"text": content})
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

        final_response = None

        if hallucination_response == 'yes':
            agent_manager = IntrospectiveAgentManager(llm_model="llama3-70b-8192", embed_model_name="BAAI/bge-small-en-v1.5")
            introspective_agent = agent_manager.create_introspective_agent(verbose=True)
            combined_context = combined_query
            combined_context += "\n".join([ctx['text'] for ctx in retrieved_contexts])
            try:
                response = introspective_agent.chat(combined_context)
            except Exception as e:
                response = introspective_agent.chat(combined_context)
            final_response = response['output']
        else:
            final_response = answer['output']

        output_folder = f"Recommendation/{file}"
        os.makedirs(output_folder, exist_ok=True)

        output_file = os.path.join(output_folder, f"{key}.txt")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_response)
    else:
        query_transformer = QueryRewriter()
        modified_query = query_transformer.apply_hyde(text)
        pipeline(modified_query, content, file, key)
    
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_content = ""
    sections = {
        "title": "",
        "abstract": "",
        "introduction": "",
        "related work": "",
        "dataset": "",
        "methodology": "",
        "method": "",
        "experimental results": "",
        "results": "",
        "discussion": "",
        "conclusion": "",
        "references": ""
    }
    
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    
    for block in blocks[:3]:  # Usually title is in first few blocks
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    # Check if text is bold (flags & 2^4 indicates bold text)
                    if span["flags"] & 16 and not any(word in span["text"].lower() for word in ["abstract", "introduction"]):
                        sections["title"] += span["text"] + " "
    
    sections["title"] = sections["title"].strip()
    print(f"Extracted title: {sections['title']}")
    
    # Get all text from the PDF with better formatting preservation
    for page in doc:
        text_content += page.get_text("text", sort=True) + "\n"
    
    # Split text into lines while preserving formatting
    lines = text_content.split('\n')
    current_section = None
    section_content = ""
    
    # Common section header patterns
    section_patterns = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "1. introduction", "i. introduction", "1 Introduction"],
        "related work": ["related work", "background", "2. related work", "ii. related work"],
        "dataset": ["dataset", "data", "2. dataset", "ii. dataset"],
        "methodology": ["methodology", "method", "proposed method", "3. methodology", "iii. methodology", "approach"],
        "experimental results": ["experimental results", "experiments", "4. experiments", "iv. experiments"],
        "results": ["results", "4. results", "iv. results"],
        "discussion": ["discussion", "5. discussion", "v. discussion"],
        "conclusion": ["conclusion", "conclusions", "6. conclusion", "vi. conclusion"],
        "references": ["references", "bibliography"]
    }
    
    # Process each line with better section detection
    for i, line in enumerate(lines):
        line = line.strip()
        line_lower = line.lower()
        
        # Check if this line is a section header
        is_header = False
        next_section = None
        
        for section, patterns in section_patterns.items():
            if any(pattern == line_lower or pattern == line_lower.strip('0123456789. ') for pattern in patterns):
                if current_section:
                    # Save the content of the previous section
                    sections[current_section] += section_content.strip() + "\n"
                current_section = section
                section_content = ""
                is_header = True
                break
            
            # Look ahead for next section to determine section boundaries
            if i < len(lines) - 1:
                next_line_lower = lines[i + 1].lower().strip()
                if any(pattern == next_line_lower or pattern == next_line_lower.strip('0123456789. ') for pattern in patterns):
                    next_section = section
        
        # Add content to current section if we're in one and it's not a header
        if current_section and not is_header:
            if line.strip():  # Only add non-empty lines
                section_content += line + " "
        
        # If we've found the next section header, save current section content
        if next_section and current_section:
            sections[current_section] += section_content.strip() + "\n"
            section_content = ""
    
    # Add the last section's content
    if current_section and section_content:
        sections[current_section] += section_content.strip()
    
    # Clean up the sections
    for section in sections:
        if section != "title":  # Don't process title section
            # Remove section headers from content
            for pattern in section_patterns.get(section, []):
                sections[section] = sections[section].replace(pattern, "")
            # Clean up extra whitespace while preserving paragraphs
            sections[section] = ' '.join(sections[section].split())
    
    return sections

dataset_folder = "Papers"



def recommend(index):
    target_file_name = f"P{index:03}.pdf"  # Format the file name to match P001, P002, etc.
    for file_name in os.listdir(dataset_folder):
        if file_name == target_file_name:  # Only process the target file
            file_path = os.path.join(dataset_folder, file_name)
            paper = extract_text_from_pdf(file_path)
            file = file_name.removesuffix('.pdf')

            # Ensure the output folder exists
            context_file = f"GDrive/{file}"

            for key, value in paper.items():
                if key == 'title' or key == 'experimental results' or key == 'methodology':
                    query_file = os.path.join(context_file, f"{key}.txt")
                    file_content = ''
                    with open(query_file, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    pipeline(value, file_content, file, key)

            break  # Exit the loop after processing the target file

recommend(4)