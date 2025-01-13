import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
import time
from tenacity import retry, stop_after_attempt, wait_exponential
from pathlib import Path
import pandas as pd
from pdf_parser import extract_text_from_pdf

def initialize_model():
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0,
        max_retries=15,
        timeout=30
    )

    # Initialize SentenceTransformer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2").to(device)
    return llm, retriever

def chunk_text(text, max_tokens=15000):
    """Split text into smaller chunks"""
    sentences = text.split('. ')
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length > max_tokens:
            chunks.append('. '.join(current_chunk) + '.')
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append('. '.join(current_chunk) + '.')
    return chunks

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def get_evaluation(llm, prompt):
    """Get evaluation with retry mechanism"""
    return llm.invoke(prompt)

def compute_similarity(retriever, text1, text2):
    """Compute semantic similarity between texts"""
    embedding1 = retriever.encode(text1, convert_to_tensor=True, device=retriever.device)
    embedding2 = retriever.encode(text2, convert_to_tensor=True, device=retriever.device)
    similarity = torch.nn.functional.cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0))
    return similarity.item()

def extract_score(evaluation_response):
    """Extract score from evaluation response"""
    try:
        evaluation_text = evaluation_response.content
        
        if "Average score:" in evaluation_text:
            score_part = evaluation_text.split("=")[-1]
            score = float(score_part.split("/")[0].strip())
            return score
        
        scores = []
        for line in evaluation_text.split('\n'):
            if "Score:" in line:
                score = float(line.split("Score:")[-1].split("/")[0].strip())
                scores.append(score)
        
        if scores:
            return sum(scores) / len(scores)
            
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None

def evaluate_paper(llm, retriever, sections, paper_title):
    """Evaluate paper sections"""
    results = {}
    
    for section_name, section_text in sections.items():
        if not section_text.strip() or section_name.lower() == "title":
            continue
            
        print(f"\nEvaluating {section_name}...")
        
        # Split large sections into chunks
        chunks = chunk_text(section_text) if len(section_text) > 4000 else [section_text]
        chunk_scores = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)} of {section_name}...")
            
            evaluation_prompt = f"""
            Evaluate this part of the {section_name} section based on:
            1. Clarity (0-100): Is the content well-structured and easy to understand?
            2. Relevance (0-100): Does it align with the main research topic?
            3. Rigor (0-100): Are the arguments and methods scientifically sound?

            Your response should look like:
            Clarity: <Score>/100
            Relevance: <Score>/100
            Rigor: <Score>/100

            Average score: (<Clarity> + <Relevance> + <Rigor>) / 3 = <Final>/100

            Section Text:
            {chunk}
            """
            
            try:
                evaluation = get_evaluation(llm, evaluation_prompt)
                time.sleep(2)  # Rate limiting
                
                score = extract_score(evaluation)
                if score is not None:
                    chunk_scores.append(score)
                    print(f"Chunk {i+1} score: {score}")
                
            except Exception as e:
                print(f"Error processing chunk {i+1} of {section_name}: {e}")
                time.sleep(5)
                continue
        
        if chunk_scores:
            avg_score = sum(chunk_scores) / len(chunk_scores)
            similarity = compute_similarity(retriever, section_text, paper_title)
            
            results[section_name] = {
                "evaluation_score": avg_score,
                "similarity_score": similarity * 100,
                "final_score": (avg_score + (similarity * 100)) / 2
            }
            
            print(f"{section_name} final scores:")
            print(f"Evaluation Score: {avg_score:.2f}")
            print(f"Similarity Score: {similarity * 100:.2f}")
            print(f"Final Score: {(avg_score + (similarity * 100)) / 2:.2f}")
    
    return results

def process_papers_in_folder(folder_path, output_csv):
    """
    Process all PDF papers in a folder and save results to CSV
    """
    results = []
    folder = Path(folder_path)  
    # Get all PDF files
    pdf_files = list(folder.glob("*.pdf"))
    total_files = len(pdf_files)
    
    print(f"Found {total_files} PDF files to process")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        try:
            print(f"\nProcessing paper {i}/{total_files}: {pdf_path.name}")
            
            # Extract sections from PDF
            sections = extract_text_from_pdf(str(pdf_path))
            paper_title = sections.get("title", "")
            
            if not paper_title:
                print(f"Could not extract title from {pdf_path.name}, skipping...")
                continue
            
            llm, retriever = initialize_model()
            # Evaluate paper
            evaluation_results = evaluate_paper(llm, retriever, sections, paper_title)
            
            # Calculate overall score
            if evaluation_results:
                overall_score = sum(scores['final_score'] for scores in evaluation_results.values()) / len(evaluation_results)
                is_publishable = 1 if overall_score >= 66 else 0
                
                # Store result
                result = {
                    'Paper_ID': pdf_path.name.removesuffix('.pdf'),
                    'Publishable': is_publishable
                }
                
                results.append(result)
                
                print(f"Paper: {paper_title}")
                print(f"Overall Score: {overall_score:.2f}")
                print(f"Publishable: {'Yes' if is_publishable else 'No'}")
                
                # Save intermediate results
                df = pd.DataFrame(results)
                df.to_csv(output_csv, index=False)
                print(f"Results saved to {output_csv}")
                
            else:
                print(f"No evaluation results for {pdf_path.name}")
            
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            continue
        
        # Add delay between papers
        time.sleep(5)
    
    return results