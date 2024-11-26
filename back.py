from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
import os
from pydantic import BaseModel
import logging
from langchain.text_splitter import CharacterTextSplitter
import tempfile
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import pdfplumber

load_dotenv()


# FastAPI app
app = FastAPI()

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s")

# Temporary storage for user-selected configurations
user_config = {
    "chunking_method": "method",
    "embedding_model": "model",
    "vector_db": "db",
    "llm_model": "model"
}

# Define request body model
class CodeGenerationRequest(BaseModel):
    chunking_method: str    
    embedding_model: str
    vector_db: str
    llm_model: str

# # Define chunking functions here
def fixed_size_chunk(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

def sentence_chunk(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]

def semantic_chunk(text, max_len=200):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    chunks = []
    current_chunk = []
    for sent in doc.sents:
        current_chunk.append(sent.text)
        if len(' '.join(current_chunk)) > max_len:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

def recursive_chunk(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# Code generation functions based on user input
def generate_chunking_code(method):
    if method == "Fixed-Size":
        return """
def fixed_size_chunk(text, max_words=100):
    words = text.split()
    return [' '.join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
"""
    elif method == "Sentence-Based":
        return """
import spacy
nlp = spacy.load("en_core_web_sm")

def sentence_chunk(text):
    doc = nlp(text)
    return [sent.text for sent in doc.sents]
"""
    elif method == "Semantic-Based":
        return """
import spacy
nlp = spacy.load("en_core_web_sm")

def semantic_chunk(text, max_len=200):
    doc = nlp(text)
    chunks = []
    current_chunk = []
    for sent in doc.sents:
        current_chunk.append(sent.text)
        if len(' '.join(current_chunk)) > max_len:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
"""
    elif method == "Recursive":
        return """
def recursive_chunk(text, max_tokens):
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return [text]
    chunks = []
    current_chunk = []
    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
"""
    return ""

def generate_embedding_code(model):
    if model == "Hugging Face":
        return """
from langchain.embeddings import HuggingFaceEmbeddings
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
"""
    elif model == "OpenAI":
        return """
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
"""
    return ""

def generate_vector_db_code(db):
    if db == "FAISS":
        return """
from langchain.vectorstores import FAISS
vector_store = FAISS.from_texts(documents, embeddings)
"""
    elif db == "ChromaDB":
        return """
from langchain.vectorstores import ChromaDB
vector_store = ChromaDB(embedding_function=embeddings.embed_query)
"""
    elif db == "Pinecone":
        return """
import pinecone
def __init__(self, PINECONE_API_KEY: str, index_name: str):
        # Initialize Pinecone
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create index if it doesn't exist
        if index_name not in [idx.name for idx in self.pc.list_indexes().indexes]:
            self.pc.create_index(
                name=index_name,
                dimension=384,  # BERT dimension
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )
        
        # Get the index
        self.index = self.pc.Index(index_name)
"""
    return ""

def generate_llm_code(model):
    if model == "llama":
        return """
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B", model_kwargs={"temperature": 0.7, "max_length": 512})
"""
    elif model == "OpenAI":
        return """
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7, max_tokens=512)
"""
    elif model == "Gemini":
        return """
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI

# Ensure the environment variable is set
os.getenv("GOOGLE_API_KEY")

# Configure Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

# Use the LLM for response
llm_response = llm.invoke(" ".join(chunks[:5]))
"""
    elif model == "Hugging Face":
        return """
from langchain.llms import HuggingFaceHub
llm = HuggingFaceHub(repo_id="some-model-id", model_kwargs={"temperature": 0.3, "max_length": 512})
"""
    return ""

# Endpoint to generate code
@app.post("/generate_code/")
async def generate_code(request: CodeGenerationRequest):
    try:
        logging.info(f"Request received: {request}")
        code = {
            "chunking_code": generate_chunking_code(request.chunking_method),
            "embedding_code": generate_embedding_code(request.embedding_model),
            "vector_db_code": generate_vector_db_code(request.vector_db),
            "llm_code": generate_llm_code(request.llm_model),
        }
        if not all(code.values()):
            raise ValueError("One or more configurations are invalid.")
        # Save the user's selected configuration
        user_config["chunking_method"] = request.chunking_method
        user_config["embedding_model"] = request.embedding_model
        user_config["vector_db"] = request.vector_db
        user_config["llm_model"] = request.llm_model

        logging.info(f"User configuration saved: {user_config}")
        return JSONResponse(content=code)
    except Exception as e:
        logging.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        logging.info(f"User configuration saved: {user_config}")
        return JSONResponse(content=code)
    except Exception as e:
        logging.error(f"Error generating code: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to process PDF using stored configuration
@app.post("/process_pdf/")
async def process_pdf(pdf_file: UploadFile = File(...)):
    try:
        if not user_config:
            raise HTTPException(status_code=400, detail="No configuration found. Generate code first.")

        # Retrieve the stored configuration
        chunking_method = user_config["chunking_method"]
        embedding_model = user_config["embedding_model"]
        vector_db = user_config["vector_db"]
        llm_model = user_config["llm_model"]
        logging.info(f"Using configuration: {user_config}")

        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            temp_pdf.write(await pdf_file.read())
            temp_pdf_path = temp_pdf.name

        # Extract content from PDF
        full_text = ""
        tables = []
        images = []
        with pdfplumber.open(temp_pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                text = page.extract_text()
                if text:
                    full_text += text + "\n"
                
                # Extract tables
                page_tables = page.extract_tables()
                if page_tables:
                    tables.extend(page_tables)
                
                # Extract images
                for img in page.images:
                    x0, y0, x1, y1 = img["x0"], img["y0"], img["x1"], img["y1"]
                    cropped_img = page.within_bbox((x0, y0, x1, y1)).to_image()
                    images.append(cropped_img)

        logging.info("PDF content extracted successfully (text, tables, images).")

        # Chunking the text
        if chunking_method == "Fixed-Size":
            chunks = fixed_size_chunk(full_text)
        elif chunking_method == "Sentence-Based":
            chunks = sentence_chunk(full_text)
        elif chunking_method == "Semantic-Based":
            chunks = semantic_chunk(full_text)
        elif chunking_method == "Recursive":
            chunks = recursive_chunk(full_text, max_tokens=100)
        else:
            raise ValueError("Invalid chunking method selected.")
        logging.info(f"Text chunked using {chunking_method}: {len(chunks)} chunks created.")

        # Generate embeddings for each chunk
        embeddings = None
        if embedding_model == "Hugging Face":
            from langchain.embeddings import HuggingFaceEmbeddings
            embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            embeddings = [embeddings_model.embed_query(chunk) for chunk in chunks]
        elif embedding_model == "OpenAI":
            from langchain.embeddings import OpenAIEmbeddings
            embeddings_model = OpenAIEmbeddings()
            embeddings = [embeddings_model.embed_query(chunk) for chunk in chunks]
        else:
            raise ValueError("Invalid embedding model selected.")
        logging.info(f"Embeddings generated using {embedding_model}.")

        # Store embeddings in vector database
        if vector_db == "FAISS":
            from langchain.vectorstores import FAISS
            vector_store = FAISS.from_texts(chunks, embeddings_model)
        elif vector_db == "ChromaDB":
            from langchain.vectorstores import Chroma
            vector_store = Chroma(embedding_function=embeddings_model.embed_query)
        elif vector_db == "Pinecone":
                import pinecone
                def __init__(self, PINECONE_API_KEY: str, index_name: str):
                    """Initialize the RAG system with Pinecone credentials"""
                    # Initialize Pinecone
                    self.pc = Pinecone(api_key=PINECONE_API_KEY)
                    
                    # Create index if it doesn't exist
                    if index_name not in [idx.name for idx in self.pc.list_indexes().indexes]:
                        self.pc.create_index(
                            name=index_name,
                            dimension=384,  # BERT dimension
                            metric='cosine',
                            spec=ServerlessSpec(
                                cloud='aws',
                                region='us-east-1'
                            )
                        )
                    
                    # Get the index
                    self.index = self.pc.Index(index_name)
        else:
            raise ValueError("Invalid vector database selected.")
        logging.info(f"Vector embeddings stored in {vector_db}.")

        # Use LLM for post-processing
        llm_response = None
        if llm_model == "llama":
            from langchain.llms import HuggingFaceHub
            llm = HuggingFaceHub(repo_id="meta-llama/Llama-3.2-1B", model_kwargs={"temperature": 0.3, "max_length": 512})
            llm_response = llm(" ".join(chunks[:5]))  # Example: Process first few chunks
        elif llm_model == "OpenAI":
            from langchain.llms import OpenAI
            llm = OpenAI(temperature=0.3, max_tokens=512)
            llm_response = llm(" ".join(chunks[:5]))
        elif llm_model == "Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.3)
            llm_response = llm.invoke(" ".join(chunks[:5]))
        else:
            raise ValueError("Invalid LLM model selected.")
        logging.info(f"LLM processed chunks using {llm_model}.")

        # Clean up the temporary file
        os.remove(temp_pdf_path)

        # Prepare the response
        response = {
            # "chunks": chunks,
            # "embeddings_status": "Successfully generated and stored.",
            "llm_response": llm_response,
            # "tables": tables,
            # "images_extracted": len(images),
        }

        return response

    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
