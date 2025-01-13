import pandas as pd
from is_publishable_checker import process_papers_in_folder
import os

def main():
    # Configure paths
    input_folder = "your input folder path"
    output_csv = "paper_evaluation_results.csv"
    
    # Process papers
    results = process_papers_in_folder(input_folder, output_csv)
    
    # Create final DataFrame and save
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
        

if __name__ == "__main__":
    os.environ['GROQ_API_KEY'] = "your groq api key"
    main()