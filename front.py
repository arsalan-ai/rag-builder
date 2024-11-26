# import streamlit as st
# import requests
# from io import BytesIO

# # FastAPI backend URL
# API_URL = "http://127.0.0.1:8000"

# # App Title
# st.title("RAG Builder")

# # Step 1: UI for selections and PDF upload
# st.header("1. Configure Your RAG Pipeline")

# # Dropdowns for chunking, embeddings, vector DB, and LLM
# chunking_method = st.selectbox("Select Chunking Method", ["Fixed-Size", "Sentence-Based", "Semantic-Based", "Recursive"])
# embedding_model = st.selectbox("Select Embedding Model", ["OpenAI", "Hugging Face"])
# vector_db = st.selectbox("Select Vector DB", ["FAISS", "ChromaDB", "Pinecone"])
# llm_model = st.selectbox("Select LLM Model", ["OpenAI", "Hugging Face", "Gemini"])

# # Submit button to generate code
# if st.button("Generate Code"):
#     st.info("Generating code based on your selections...")
#     try:
#         response = requests.post(
#             f"{API_URL}/generate_code/",
#             json={
#                 "chunking_method": chunking_method,
#                 "embedding_model": embedding_model,
#                 "vector_db": vector_db,
#                 "llm_model": llm_model,
#             },
#         )

#         if response.status_code == 200:
#             code = response.json()
#             st.code(code.get("chunking_code", "# Error: Missing chunking code"), language="python")
#             st.code(code.get("embedding_code", "# Error: Missing embedding code"), language="python")
#             st.code(code.get("vector_db_code", "# Error: Missing vector DB code"), language="python")
#             st.code(code.get("llm_code", "# Error: Missing LLM code"), language="python")
#         else:
#             st.error(f"Error: {response.json().get('error', 'Unknown error occurred')}")
#     except Exception as e:
#         st.error(f"Failed to connect to backend: {e}")

# # Step 2: Upload PDF files for processing
# st.header("2. Upload PDFs for Processing")
# uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# if uploaded_files:
#     st.write(f"Uploaded {len(uploaded_files)} file(s).")
#     process_button = st.button("Process PDFs")

#     if process_button:
#         st.info("Processing PDFs...")
#         try:
#             for file in uploaded_files:
#                 # Read the file and send it to the backend
#                 file_data = {"file": (file.name, file, "application/pdf")}
#                 response = requests.post(
#                     f"{API_URL}/process_pdf/",
#                     files=file_data,
#                     data={"chunking_method": chunking_method},
#                 )

#                 if response.status_code == 200:
#                     result = response.json()
#                     st.write(f"Chunks for {file.name}:")
#                     st.write(result.get("chunks", []))
#                 else:
#                     st.error(f"Error processing {file.name}: {response.json().get('detail', 'Unknown error')}")
#         except Exception as e:
#             st.error(f"Failed to process PDFs: {e}")

# # Step 3: User Query
# st.header("3. Ask a Question")
# query = st.text_input("Enter your query:")

# if query and st.button("Submit Query"):
#     st.info("Submitting query...")
#     # Placeholder for query integration
#     try:
#         response = requests.post(
#             f"{API_URL}/query/",  # Replace with your actual query endpoint
#             json={"query": query},
#         )

#         if response.status_code == 200:
#             result = response.json()
#             st.write("Answer:")
#             st.write(result.get("answer", "No answer returned."))
#         else:
#             st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
#     except Exception as e:
#         st.error(f"Failed to connect to backend: {e}")

import streamlit as st
import requests
import json

# Set backend API URL
API_BASE_URL = "http://localhost:8000"  # Replace with the actual backend URL if different

# Streamlit app title
st.title("RAG Builder Frontend")

# Sidebar for mode selection
mode = st.sidebar.selectbox("Select Mode", ["Generate Code", "Process PDF"])

# Generate Code Mode
if mode == "Generate Code":
    st.header("Generate Code for RAG Pipeline")

    # Form to collect user input for code generation
    with st.form("code_generation_form"):
        chunking_method = st.selectbox(
            "Select Chunking Method",
            ["Fixed-Size", "Sentence-Based", "Semantic-Based", "Recursive"],
        )
        embedding_model = st.selectbox(
            "Select Embedding Model",
            ["Hugging Face", "OpenAI"],
        )
        vector_db = st.selectbox(
            "Select Vector Database",
            ["FAISS", "ChromaDB", "Pinecone"],
        )
        llm_model = st.selectbox(
            "Select LLM Model",
            ["llama", "OpenAI", "Gemini", "Hugging Face"],
        )
        submit_button = st.form_submit_button("Generate Code")

    # On form submission, make a request to the backend
    if submit_button:
        payload = {
            "chunking_method": chunking_method,
            "embedding_model": embedding_model,
            "vector_db": vector_db,
            "llm_model": llm_model,
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/generate_code/", json=payload
            )
            if response.status_code == 200:
                code_snippets = response.json()
                st.success("Code generated successfully!")
                st.code(json.dumps(code_snippets, indent=4), language="json")
            else:
                st.error(f"Error: {response.json()['detail']}")
        except Exception as e:
            st.error(f"Error connecting to backend: {e}")

# Process PDF Mode
elif mode == "Process PDF":
    st.header("Process PDF Document")

    # File uploader for PDF
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # Process the uploaded file
    if uploaded_file is not None:
        with st.spinner("Processing the PDF..."):
            try:
                # Send the file to the backend
                files = {"pdf_file": uploaded_file.getvalue()}
                response = requests.post(
                    f"{API_BASE_URL}/process_pdf/", files={"pdf_file": uploaded_file}
                )

                if response.status_code == 200:
                    result = response.json()
                    st.success("PDF processed successfully!")

                    # Display the chunks
                    st.subheader("Extracted Text Chunks")
                    for chunk in result.get("chunks", []):
                        st.write(chunk)

                    # Display the tables
                    st.subheader("Extracted Tables")
                    for table in result.get("tables", []):
                        st.write(table)

                    # Display the number of extracted images
                    st.subheader("Extracted Images")
                    st.write(f"Number of Images Extracted: {result.get('images_extracted', 0)}")

                    # Display LLM response
                    st.subheader("LLM Response")
                    st.write(result.get("llm_response", "No response"))

                else:
                    st.error(f"Error: {response.json()['detail']}")
            except Exception as e:
                st.error(f"Error connecting to backend: {e}")
