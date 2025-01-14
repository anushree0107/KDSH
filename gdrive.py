import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import fitz
from Retriever.connector import GDriveRulebook

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
    
    # Extract title from first page
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]
    
    # Look for title in the first few blocks
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

dataset_folder = "/home/anushree/KDSH/Papers"

gdrive = GDriveRulebook()

def load_research_papers(folder_path):
    papers = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            paper = extract_text_from_pdf(file_path)
            file = file_name.removesuffix('.pdf')
            papers[file_name] = paper

            # Ensure the output folder exists
            output_folder = f"/home/anushree/KDSH/GDrive/{file}"
            os.makedirs(output_folder, exist_ok=True)  # Create folder if it does not exist

            for key, value in paper.items():
                response = gdrive.query_vector_store(value)
                s=''
                for i in response:
                    s+=i["text"]
                output_file = os.path.join(output_folder, f"{key}.txt")
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(s)

load_research_papers(dataset_folder)