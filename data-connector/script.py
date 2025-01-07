import pathway as pw

# Define the Google Drive connector
table = pw.io.gdrive.read(
    object_id="10Tn2WmWaYkysgaEsGcC1k-vcTbLNb5QY",
    service_user_credentials_file="credentials.json",
    mode="streaming",
    with_metadata=True,
    file_name_pattern="*.pdf"  # Only process PDFs
)

# Select metadata and content from the table
def process_files(table: pw.Table):
    return table.select(
        file_id=table._metadata["id"],         # File ID
        file_name=table._metadata["name"],     # File name
        file_type=table._metadata["mimeType"], # MIME type
        content=table.data                     # Binary content
    )

processed_table = process_files(table)
print(processed_table)