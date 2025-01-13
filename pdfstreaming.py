from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import pdfplumber  # Use pdfplumber to read the streamed PDF data

# Define the scope and load credentials
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
credentials = Credentials.from_service_account_file(
    '/mnt/c/Users/HP/OneDrive/Desktop/kdsh-task-2/KDSH/Retriever/credentials.json', 
    scopes=SCOPES
)

# Build the Google Drive API service
service = build('drive', 'v3', credentials=credentials)

# Function to list files in a folder (recursive for subfolders)
def list_files_in_folder(service, folder_id):
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name, mimeType)").execute()
    files = results.get('files', [])
    pdf_files = []
    for file in files:
        if file['mimeType'] == 'application/pdf':  # Check if it's a PDF
            pdf_files.append(file)
        elif file['mimeType'] == 'application/vnd.google-apps.folder':  # If subfolder
            pdf_files.extend(list_files_in_folder(service, file['id']))
    return pdf_files

# Function to stream a PDF file and read its content
def stream_pdf_file(service, file_id):
    request = service.files().get_media(fileId=file_id)
    pdf_stream = io.BytesIO()  # Create an in-memory file-like object
    downloader = MediaIoBaseDownload(pdf_stream, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Streaming progress: {int(status.progress() * 100)}% complete.")
    
    pdf_stream.seek(0)  # Reset the stream position to the beginning
    return pdf_stream

# Folder ID from Google Drive
folder_id = "1zY8vrCXC2ore-wuNGDVZCAqCggS1FIOB"  # Replace with your folder ID

# List PDFs in the folder
pdf_files = list_files_in_folder(service, folder_id)

for pdf in pdf_files:
    print(f"Streaming: {pdf['name']}")
    pdf_stream = stream_pdf_file(service, pdf['id'])
    
    # Example: Read the first 500 characters using pdfplumber
    # Example: Read the first 500 characters using pdfplumber with fixed spaces
    with pdfplumber.open(pdf_stream) as pdf_reader:
        content = ""
        for page in pdf_reader.pages:
            # Use 'page.extract_text(x_tolerance=2, y_tolerance=2)' to better handle spaces
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text:
                content += page_text
            # if len(content) >= 500:
            #     break  # Stop if we have 500 characters
        print(f"Content of {pdf['name']}:\n{content}")