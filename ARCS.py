import streamlit as st
import fitz  # PyMuPDF
from final_pipeline import pipeline
from Retriever.emnlp_retriever import EMNLPRulebook

# Function to extract text and sections from a PDF
def extract_text_from_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
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

    # Extract title from the first page
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    for block in blocks[:3]:  # Title is often in the first few blocks
        if "lines" in block:
            for line in block["lines"]:
                for span in line["spans"]:
                    if span["flags"] & 16 and not any(word in span["text"].lower() for word in ["abstract", "introduction"]):
                        sections["title"] += span["text"] + " "

    sections["title"] = sections["title"].strip()

    # Get all text from the PDF
    for page in doc:
        text_content += page.get_text("text", sort=True) + "\n"

    # Split text into lines
    lines = text_content.split('\n')
    current_section = None
    section_content = ""

    # Section patterns
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

    for i, line in enumerate(lines):
        line = line.strip()
        line_lower = line.lower()
        is_header = False
        next_section = None

        for section, patterns in section_patterns.items():
            if any(pattern == line_lower or pattern == line_lower.strip('0123456789. ') for pattern in patterns):
                if current_section:
                    sections[current_section] += section_content.strip() + "\n"
                current_section = section
                section_content = ""
                is_header = True
                break

            if i < len(lines) - 1:
                next_line_lower = lines[i + 1].lower().strip()
                if any(pattern == next_line_lower or pattern == next_line_lower.strip('0123456789. ') for pattern in patterns):
                    next_section = section

        if current_section and not is_header:
            if line.strip():
                section_content += line + " "

        if next_section and current_section:
            sections[current_section] += section_content.strip() + "\n"
            section_content = ""

    if current_section and section_content:
        sections[current_section] += section_content.strip()

    for section in sections:
        if section != "title":
            for pattern in section_patterns.get(section, []):
                sections[section] = sections[section].replace(pattern, "")
            sections[section] = ' '.join(sections[section].split())

    return sections

# Streamlit App
st.title("Research Paper Evaluation and Conference Recommendation")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file is not None:
    print(uploaded_file.name)
    st.write(f"Uploaded file: {uploaded_file.name}")

    if st.button("Evaluate"):
        try:
            extracted_sections = extract_text_from_pdf(uploaded_file)

            query_list = []

            for section, content in extracted_sections.items():
                query_list.append({
                    "section": section,
                    "content": content
                })

            retriver = EMNLPRulebook()
            retrived_contexts = []
            for query in query_list:
                retrived_context = retriver.query_vector_store(query["content"])
                retrived_contexts.append(retrived_context)

            query = query_list[0]["content"]
            retrieved_context = retriver.query_vector_store(query)

            # for i in range(len(query_list)):
            final_conference, rationale = pipeline(retriver, query, retrieved_context)
            st.success("PDF evaluated successfully!")
            st.write("### Recommended Conference: ", final_conference)
            st.write("**Rationale**: ", rationale)

        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a PDF file to proceed.")