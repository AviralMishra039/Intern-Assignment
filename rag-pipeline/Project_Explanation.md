# Comprehensive Guide: RAG Pipeline Project

This document thoroughly explains every technological choice, architectural decision, parameter setting, and line-of-code rationale behind the Retrieval-Augmented Generation (RAG) system you built. Use this guide to prepare for any technical interview questions about your project!

---

## 1. The Core Architecture: How RAG Works
RAG (Retrieval-Augmented Generation) prevents an AI from hallucinating by forcing it to read relevant documents *before* answering a question. The architecture is divided into two distinct flows:

### Flow A: Ingestion (Uploading Documents)
1. **Upload**: User sends a PDF to the FastAPI endpoint.
2. **Parsing**: We extract the raw text from the PDF.
3. **Chunking**: We split the long text into small, readable paragraphs ("chunks").
4. **Embedding**: We send each chunk to OpenAI to convert the text into a mathematical vector (an array of numbers representing the meaning of the text).
5. **Storage**: We store the metadata in SQLite, and we store the text chunks + vectors in ChromaDB.

### Flow B: Retrieval & Generation (Answering Questions)
1. **Question**: User asks a question.
2. **Embed Query**: We convert the user's question into a vector using the exact same OpenAI model.
3. **Vector Search (Retrieval)**: We query ChromaDB to find the vectors (chunks) mathematically closest to the question's vector.
4. **Prompting (Generation)**: We give the retrieved chunks and the user's question to the LLM (`gpt-4o-mini`) and instruct it to answer *only* based on the chunks provided.

---

## 2. Technology Stack & "The Why"

| Technology | Role | Why we chose it |
| :--- | :--- | :--- |
| **FastAPI** | API Framework | It is modern, asynchronous, incredibly fast, and automatically generates the Swagger UI (`/docs`) for testing. It is the industry standard for Python microservices. |
| **ChromaDB** | Vector Database | It can run in "embedded mode" (locally in files) meaning we don't have to spin up a separate database server. It is open-source and natively built for Python. |
| **SQLite (via SQLAlchemy)** | Metadata Database | Relational databases are best for managing structured data (IDs, upload times, page counts). SQLite is file-based, requiring zero extra infrastructure. |
| **OpenAI `text-embedding-3-small`** | Embedding Model | It is OpenAI's newest, cheapest, and most efficient embedding model. It captures semantic meaning better than older NLP models. |
| **OpenAI `gpt-4o-mini`** | Large Language Model | It is blazing fast, wildly cheap, has a massive 128k token context window, and is smart enough to summarize highly complex documentation. |
| **PyMuPDF (`fitz`)** | PDF Parser | It is mathematically proven to be one of the fastest and most accurate PDF text extractors available in Python (much better than `PyPDF2`). |
| **LangChain Text Splitter** | Chunking Utility | It intelligently splits text by looking for paragraphs, then sentences, then words. This prevents a sentence from being chopped exactly in half. |
| **Docker** | Containerization | Ensures the app runs perfectly on *any* computer or cloud server without "it works on my machine" dependency nightmares. |

---

## 3. Parameter Decisions

### Text Chunking
* **`CHUNK_SIZE = 500` characters**: We chunked the text into roughly 500-character blocks (about 100 words). If chunks are too small (e.g., 50 chars), the AI lacks context. If they are too large (e.g., 5000 chars), the vector search becomes blurry because the chunk contains too many varying topics.
* **`CHUNK_OVERLAP = 50` characters**: When chopping up a PDF, a clean chunk split might accidentally cut a sentence right in the middle. Having an overlap ensures that the end of Chunk 1 is partially repeated at the beginning of Chunk 2, preserving context flow.

### Retrieval
* **`TOP_K_RESULTS = 40`**: When the user asks a question, we tell ChromaDB to return the top 40 most relevant chunks. Because we allow the user to query up to 20 PDFs at once, setting this number high ensures the system casts a wide net. Since `gpt-4o-mini` has a huge context window, processing 40 chunks (~20,000 characters) is effortless.

---

## 4. Code Structure: What Each File Does

### `/app/main.py` (The Entry Point)
* **What it does**: Initializes the FastAPI app, sets up the lifespan to create the database tables on startup, configures CORS (so frontend apps can talk to it), and includes the routers.
* **Why**: Keeps the application bootstrapping logic isolated.

### `/app/config.py` (The Settings)
* **What it does**: Uses a `Settings` class to read the `.env` file and set defaults.
* **Why**: Security and cleanliness. We NEVER hardcode API keys in the app. Everywhere else in the app imports config variables through this file.

### `/app/database.py` (The State)
* **What it does**: Connects to the SQLite file using SQLAlchemy and sets up the ChromaDB persistent client. Contains the `Document` schema model.
* **Why**: Centralizes database connections so we don't open 50 redundant connections simultaneously.

### `/app/routers/` (The Endpoints)
* `documents.py`: Handles the `POST` upload logic. Checks if the file is valid, ensures they haven't exceeded 20 documents, saves the file to disk, sends it to the processing service, and deletes the temporary file.
* `query.py`: Receives the question, validates it isn't empty, runs the retrieval service, runs the LLM service, and packages the answer + citations.
* **Why**: In FastAPI, "routers" act as the controllers holding the API definitions. They delegate heavy lifting to "services".

### `/app/services/` (The Heavy Lifting)
* `ingestion.py`: The heart of data processing. Reads PDFs, chunks them, calls the embedding API, and stores them in ChromaDB.
* `retrieval.py`: Takes a question, embeds it, and runs the vector math to pull out the chunks.
* `llm.py`: Glues the retrieved chunks together into a prompt block, attaches the system instructions ("answer only based on context"), and calls OpenAI.
* **Why**: Separating business logic from API endpoints makes the code modular, readable, and infinitely easier to test.

### `/app/models/schemas.py` (The Validators)
* **What it does**: Defines Pydantic models for exactly what JSON data a user is required to send, and exactly what JSON the API will return.
* **Why**: FastAPI automatically validates incoming data against these schemas. If a user sends a number instead of a string for a question, Pydantic immediately throws an error without crashing the server.

### `/tests/` (The Safety Net)
* **The Strategy**: We wrote tests that **Mock** (fake) the OpenAI API.
* **Why**: If tests rely on real OpenAI requests, you burn money every time you run tests, and tests will fail without an internet connection. Testing should verify *your* logic, not OpenAI's servers.

### `Dockerfile`
* **What it does**: Uses a `slim` Debian Linux image, installs Python, installs SQLite, and copies our code.
* **Why**: The `slim` version ensures our Docker image is small (megabytes, not gigabytes), making cloud deployments much faster.
