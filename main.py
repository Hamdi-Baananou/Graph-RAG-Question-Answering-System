# =============================
# Required imports
# =============================
import os
import re
import requests
import json
import gradio as gr
import logging
import time
from datetime import datetime
import pandas as pd
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# RAG imports
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Graph DB imports
from langchain_community.graphs import Neo4jGraph
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.chains import GraphQAChain
#from langchain_community.llms import OpenRouter
# Replace the import
# from langchain_community.llms import OpenRouter
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional
import requests

# Create a custom OpenRouter LLM class
class OpenRouter(LLM):
    api_key: str
    model: str
    max_tokens: int = 256
    temperature: float = 0.1
    openrouter_api_base: str = "https://openrouter.ai/api/v1/chat/completions"
    
    @property
    def _llm_type(self) -> str:
        return "openrouter"
    
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://colab.research.google.com",
            "X-Title": "Graph RAG Assistant"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if stop:
            payload["stop"] = stop
            
        response = requests.post(self.openrouter_api_base, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise ValueError(f"API Error ({response.status_code}): {response.text}")
            
        response_data = response.json()
        
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            raise ValueError("Invalid response format from API")
            
        first_choice = response_data['choices'][0]
        message_content = first_choice.get('message', {}).get('content', '')
        
        return message_content
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }

# =============================
# Setup Logging
# =============================
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output for different log levels"""
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m\033[1m',  # Bold Red
        'ENDC': '\033[0m',       # Reset color
    }
    
    def format(self, record):
        log_message = super().format(record)
        return f"{self.COLORS.get(record.levelname, '')}{log_message}{self.COLORS['ENDC']}"

# Setup logger
logger = logging.getLogger('graph_rag')
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
console_format = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# File handler for more permanent logging
file_handler = logging.FileHandler('graph_rag.log')
file_handler.setLevel(logging.DEBUG)
file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_format)
logger.addHandler(file_handler)

# =============================
# OpenRouter API Configuration
# =============================
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL_NAME = "google/gemma-3-27b-it:free"

# =============================
# Neo4j Configuration
# =============================
# These will be collected from the UI
NEO4J_URI = "bolt://[colab-instance-ip]:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "mypassword"

# =============================
# Initialize Embeddings (Using Hugging Face)
# =============================
def get_embeddings():
    logger.info("Initializing embedding model")
    start_time = time.time()
    embedding_function = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    logger.info(f"Embedding model loaded in {time.time() - start_time:.2f} seconds")
    return embedding_function

# =============================
# PDF Processing Functions
# =============================
def process_pdfs(pdf_files):
    """Process PDFs with error handling, text cleaning, and chunking for graph knowledge base"""
    logger.info(f"Starting to process {len(pdf_files)} PDF files")
    all_chunks = []
    source_metadata = {}
    
    if not pdf_files:
        logger.error("No PDF files provided")
        raise ValueError("No PDF files provided")

    for pdf_file in pdf_files:
        try:
            start_time = time.time()
            logger.info(f"Processing {pdf_file.name}")
            
            # Load PDF
            loader = PyMuPDFLoader(pdf_file.name)
            documents = loader.load()
            logger.debug(f"Loaded {len(documents)} pages from {pdf_file.name}")
            
            # Store metadata about the source document
            source_name = os.path.basename(pdf_file.name)
            source_metadata[source_name] = {
                "filename": source_name,
                "pages": len(documents),
                "processed_at": datetime.now().isoformat()
            }
            
            # Clean text
            for doc in documents:
                doc.page_content = re.sub(r'\s+', ' ', doc.page_content).strip()
                doc.metadata['source'] = source_name
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            chunks = text_splitter.split_documents(documents)
            
            # Add more detailed metadata to chunks
            for i, chunk in enumerate(chunks):
                chunk.metadata['chunk_id'] = f"{source_name}_chunk_{i}"
                chunk.metadata['source_document'] = source_name
            
            all_chunks.extend(chunks)
            logger.info(f"Processed {pdf_file.name} into {len(chunks)} chunks in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}", exc_info=True)
            continue

    logger.info(f"Completed processing all PDFs. Generated {len(all_chunks)} total chunks")
    return all_chunks, source_metadata

# =============================
# Neo4j Graph Functions
# =============================
def connect_to_neo4j(uri, username, password):
    """Establish connection to Neo4j and return graph object"""
    try:
        logger.info(f"Connecting to Neo4j at {uri}")
        graph = Neo4jGraph(
            url=uri,
            username=username, 
            password=password
        )
        # Test connection
        result = graph.query("MATCH (n) RETURN count(n) as count")
        logger.info(f"Successfully connected to Neo4j. Database has {result[0]['count']} nodes")
        return graph
    except Exception as e:
        logger.error(f"Failed to connect to Neo4j: {str(e)}", exc_info=True)
        raise

def setup_neo4j_schema(graph):
    """Define and setup the graph schema"""
    logger.info("Setting up Neo4j schema and constraints")
    try:
        # Create constraints
        constraints = [
            "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT relationship_type IF NOT EXISTS FOR ()-[r:MENTIONS]-() REQUIRE r.type IS NOT NULL",
        ]
        
        for constraint in constraints:
            try:
                graph.query(constraint)
                logger.debug(f"Applied constraint: {constraint}")
            except Exception as e:
                logger.warning(f"Constraint already exists or failed: {str(e)}")
                continue
                
        # Create indexes
        indexes = [
            "CREATE INDEX document_source_idx IF NOT EXISTS FOR (d:Document) ON (d.source)",
            "CREATE INDEX chunk_content_idx IF NOT EXISTS FOR (c:Chunk) ON (c.content)"
        ]
        
        for index in indexes:
            try:
                graph.query(index)
                logger.debug(f"Created index: {index}")
            except Exception as e:
                logger.warning(f"Index already exists or failed: {str(e)}")
                continue
                
        logger.info("Neo4j schema setup completed")
    except Exception as e:
        logger.error(f"Error setting up Neo4j schema: {str(e)}", exc_info=True)
        raise

def create_knowledge_graph(graph, chunks, source_metadata, api_key):
    """Create a knowledge graph from the document chunks"""
    logger.info("Starting knowledge graph creation")
    try:
        # Create nodes for source documents
        for source, metadata in source_metadata.items():
            # Create document node
            logger.debug(f"Creating document node for {source}")
            query = """
            MERGE (d:Document {id: $id})
            SET d.title = $title,
                d.pages = $pages,
                d.processed_at = $processed_at
            RETURN d
            """
            params = {
                "id": metadata["filename"],
                "title": metadata["filename"],
                "pages": metadata["pages"],
                "processed_at": metadata["processed_at"]
            }
            graph.query(query, params=params)
        
        # Setup OpenRouter for entity extraction
        openrouter = OpenRouter(
            api_key=api_key,
            model=MODEL_NAME,
            max_tokens=256,
            temperature=0.1,
            openrouter_api_base=OPENROUTER_API_URL
        )
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            if i % 10 == 0:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Create chunk node
            chunk_id = chunk.metadata['chunk_id']
            source_doc = chunk.metadata['source_document']
            
            # Process chunk to extract entities
            try:
                logger.debug(f"Extracting entities from chunk {chunk_id}")
                
                entity_prompt = f"""
                Extract all important named entities, concepts, and topics from this text. 
                Return them as a JSON list of objects with "type" and "name" properties.
                Examples of entity types: Person, Organization, Product, Technology, Concept, Topic.
                
                TEXT: {chunk.page_content[:1500]}
                
                JSON RESPONSE:
                """
                
                entity_response = openrouter.invoke(entity_prompt)
                
                # Parse the JSON response
                try:
                    # Find JSON in the response using regex
                    json_match = re.search(r'\[.*\]', entity_response, re.DOTALL)
                    if json_match:
                        entities = json.loads(json_match.group(0))
                    else:
                        # Try to find JSON with curly braces
                        json_match = re.search(r'\{.*\}', entity_response, re.DOTALL)
                        if json_match:
                            potential_json = json_match.group(0)
                            entities = json.loads(f"[{potential_json}]")
                        else:
                            entities = []
                            logger.warning(f"Could not extract JSON from entity response: {entity_response[:100]}...")
                except Exception as e:
                    logger.warning(f"Failed to parse entity JSON: {str(e)}. Response: {entity_response[:100]}...")
                    entities = []
                
                # Create chunk node with its content and link to document
                logger.debug(f"Creating chunk node {chunk_id} and linking to document {source_doc}")
                query = """
                MERGE (c:Chunk {id: $chunk_id})
                SET c.content = $content,
                    c.page_num = $page_num
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:CONTAINS]->(c)
                RETURN c
                """
                params = {
                    "chunk_id": chunk_id,
                    "content": chunk.page_content,
                    "page_num": chunk.metadata.get('page', 0),
                    "doc_id": source_doc
                }
                graph.query(query, params=params)
                
                # Create entity nodes and relationships
                for entity in entities:
                    if not isinstance(entity, dict) or 'name' not in entity or 'type' not in entity:
                        logger.warning(f"Invalid entity format: {entity}")
                        continue
                        
                    entity_name = entity.get('name')
                    entity_type = entity.get('type')
                    
                    if not entity_name or not entity_type:
                        continue
                    
                    # Clean entity name and create ID
                    clean_name = re.sub(r'[^\w]', '_', entity_name).lower()
                    entity_id = f"{clean_name}_{entity_type.lower()}"
                    
                    # Create entity node and link to chunk
                    logger.debug(f"Creating entity node {entity_id} and linking to chunk {chunk_id}")
                    query = """
                    MERGE (e:Entity {id: $entity_id})
                    SET e.name = $name,
                        e.type = $type
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS {type: $type}]->(e)
                    RETURN e
                    """
                    params = {
                        "entity_id": entity_id,
                        "name": entity_name,
                        "type": entity_type,
                        "chunk_id": chunk_id
                    }
                    graph.query(query, params=params)
                
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_id}: {str(e)}", exc_info=True)
                continue
        
        # Create connections between related entities
        logger.info("Creating connections between related entities")
        query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e1:Entity)
        MATCH (c)-[:MENTIONS]->(e2:Entity)
        WHERE e1 <> e2
        MERGE (e1)-[r:RELATED_TO]->(e2)
        ON CREATE SET r.weight = 1
        ON MATCH SET r.weight = r.weight + 1
        """
        graph.query(query)
        
        logger.info("Knowledge graph creation completed")
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}", exc_info=True)
        raise

def setup_vector_index(chunks, embedding_function, neo4j_uri, neo4j_username, neo4j_password):
    """Setup vector embeddings in Neo4j for semantic search"""
    logger.info("Setting up vector index for semantic search")
    try:
        # Create vector index in Neo4j with minimal parameters
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]
        
        vector_index = Neo4jVector.from_texts(
            texts=texts,
            embedding=embedding_function,
            metadatas=metadatas,
            url=neo4j_uri,
            username=neo4j_username,
            password=neo4j_password,
            index_name="chunk_embeddings",
            node_label="Chunk"
        )
        
        logger.info(f"Vector index created with {len(chunks)} chunk embeddings")
        return vector_index
    except Exception as e:
        logger.error(f"Error setting up vector index: {str(e)}", exc_info=True)
        raise
# =============================
# Query Functions
# =============================
def get_query_context(question, vector_store, graph):
    """Get context for query using both vector similarity and graph relationships"""
    logger.info(f"Getting context for question: {question}")
    contexts = []
    
    try:
        # Step 1: Extract potential entities from the question
        start_time = time.time()
        logger.debug("Identifying key entities in the question")
        entities_in_question = extract_entities_from_question(question)
        logger.debug(f"Identified entities: {entities_in_question}")
        
        # Step 2: Use vector similarity to get relevant chunks
        logger.debug("Performing vector similarity search")
        vector_results = vector_store.similarity_search(question, k=3)
        logger.debug(f"Vector search found {len(vector_results)} relevant chunks")
        
        for idx, doc in enumerate(vector_results):
            contexts.append({
                "source": "vector_similarity",
                "rank": idx,
                "chunk_id": doc.metadata.get('chunk_id', 'unknown'),
                "content": doc.page_content,
                "source_document": doc.metadata.get('source_document', 'unknown')
            })
        
        # Step 3: Use graph relationships to find related chunks
        if entities_in_question:
            logger.debug("Performing graph traversal to find related content")
            for entity in entities_in_question:
                # Find chunks that mention this entity or related entities
                query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id, c.content as content, c.page_num as page_num
                UNION
                MATCH (e1:Entity)-[:RELATED_TO]->(e2:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE toLower(e1.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id, c.content as content, c.page_num as page_num
                LIMIT 3
                """
                graph_results = graph.query(query, params={"entity_name": entity})
                
                for idx, result in enumerate(graph_results):
                    contexts.append({
                        "source": "graph_traversal",
                        "entity": entity,
                        "rank": idx,
                        "chunk_id": result.get('chunk_id', 'unknown'),
                        "content": result.get('content', ''),
                        "page_num": result.get('page_num', 0)
                    })
        
        logger.info(f"Context gathering completed in {time.time() - start_time:.2f} seconds. Found {len(contexts)} relevant contexts")
        return contexts
    
    except Exception as e:
        logger.error(f"Error getting query context: {str(e)}", exc_info=True)
        return contexts

def extract_entities_from_question(question):
    """Simple keyword extraction for entity matching"""
    # Remove common stop words
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'about', 'like', 'through', 'over', 'before', 'after', 'since', 'of', 'from'}
    
    # Tokenize and filter
    words = re.findall(r'\b\w+\b', question.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 3]
    
    # Get noun phrases (simple approximation)
    text = question.lower()
    phrases = re.findall(r'\b[a-z]+\s+[a-z]+\b', text)
    phrases = [p for p in phrases if not any(word in stop_words for word in p.split())]
    
    # Combine single words and phrases
    entities = filtered_words + phrases
    return list(set(entities))  # Remove duplicates

def answer_question(question, api_key, contexts, neo4j_uri, neo4j_username, neo4j_password):
    """Generate answer based on retrieved contexts using OpenRouter API"""
    logger.info(f"Generating answer for question: {question}")
    start_time = time.time()
    
    try:
        # Compile context
        context_text = "\n\n".join([
            f"Source: {ctx['source']}, ID: {ctx['chunk_id']}\nContent: {ctx['content']}"
            for ctx in contexts
        ])
        
        # Add graph schema information
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        schema_text = graph.get_schema
        
        # Formulating the API prompt
        prompt = f"""You are a specialized document assistant working with a graph-based knowledge system.

        Knowledge Graph Schema:
        {schema_text}

        Context information from documents:
        {context_text}

        Question: {question}

        Answer the question based on the context provided. If you're unsure or the information isn't in the context, say "I don't have enough information in my knowledge base to answer this question properly."

        Include references to document sources when possible. Be precise and concise."""

        # Defining API request payload
        payload = {
            "model": MODEL_NAME,
            "max_tokens": 2048,
            "top_p": 1,
            "top_k": 40,
            "temperature": 0.3,
            "messages": [{"role": "user", "content": prompt}]
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://colab.research.google.com",
            "X-Title": "Graph RAG Assistant"
        }

        # Log the API request
        logger.debug(f"Sending request to OpenRouter API with payload size: {len(str(payload))}")
        
        # Sending request to OpenRouter API
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload)

        # Debug the API response
        logger.debug(f"API Response Status: {response.status_code}")
        logger.debug(f"API Response Headers: {response.headers}")
        
        if response.status_code != 200:
            logger.error(f"API Error ({response.status_code}): {response.text[:500]}...")
            return f"API Error ({response.status_code}): {response.text[:200]}..."

        response_data = response.json()
        logger.debug(f"API Response JSON: {json.dumps(response_data)[:500]}...")

        # Handle OpenRouter's response format
        if 'choices' not in response_data or len(response_data['choices']) == 0:
            logger.error("Error: Invalid response format from API")
            return "Error: Invalid response format from API"

        first_choice = response_data['choices'][0]
        message_content = first_choice.get('message', {}).get('content', '')

        if not message_content:
            logger.error("Error: Empty response from API")
            return "Error: Empty response from API"

        # Clean up the response
        cleaned_answer = re.sub(r'\\boxed{', '', message_content)  # Remove LaTeX box start
        cleaned_answer = re.sub(r'\\[^\s]+', '', cleaned_answer)   # Remove other LaTeX commands
        cleaned_answer = cleaned_answer.replace('\\n', '\n')       # Convert escaped newlines

        logger.info(f"Answer generated in {time.time() - start_time:.2f} seconds")
        return cleaned_answer.strip()

    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        return f"Error generating answer: {str(e)}"

# =============================
# Visualization Functions
# =============================
def visualize_graph(neo4j_uri, neo4j_username, neo4j_password):
    """Generate a visualization of the knowledge graph structure"""
    logger.info("Generating graph visualization")
    try:
        graph = Neo4jGraph(url=neo4j_uri, username=neo4j_username, password=neo4j_password)
        
        # Get graph statistics
        doc_count = graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
        chunk_count = graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
        entity_count = graph.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
        rel_count = graph.query("MATCH ()-[r]-() RETURN count(r) as count")[0]['count']
        
        # Get entity distribution
        entity_types = graph.query("MATCH (e:Entity) RETURN e.type as type, count(*) as count ORDER BY count DESC")
        
        # Create a sample visualization
        html = f"""
        <div style="background-color: #f5f5f5; padding: 20px; border-radius: 10px;">
            <h2>Knowledge Graph Overview</h2>
            <p><b>Documents:</b> {doc_count}</p>
            <p><b>Chunks:</b> {chunk_count}</p>
            <p><b>Entities:</b> {entity_count}</p>
            <p><b>Relationships:</b> {rel_count}</p>
            
            <h3>Entity Types</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th style="padding: 8px; text-align: left;">Entity Type</th>
                    <th style="padding: 8px; text-align: left;">Count</th>
                </tr>
        """
        
        for item in entity_types:
            html += f"""
                <tr>
                    <td style="padding: 8px;">{item['type']}</td>
                    <td style="padding: 8px;">{item['count']}</td>
                </tr>
            """
            
        html += """
            </table>
        </div>
        """
        
        # Sample subgraph for visualization
        sample_query = """
        MATCH (d:Document)-[:CONTAINS]->(c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.type = 'Concept' OR e.type = 'Topic'
        RETURN d.id as doc, e.name as entity, e.type as type
        LIMIT 20
        """
        sample_data = graph.query(sample_query)
        
        # Convert to DataFrame for plotting
        if sample_data:
            df = pd.DataFrame(sample_data)
            
            # Display the sample
            logger.debug("Generated visualization data")
            
            return HTML(html)
        else:
            return "No visualization data available"
    
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}", exc_info=True)
        return f"Error generating visualization: {str(e)}"

# =============================
# Main Process Functions
# =============================
def process_files_and_build_graph(pdf_files, neo4j_uri, neo4j_username, neo4j_password, api_key):
    """Full process from files to graph database with comprehensive logging"""
    try:
        logger.info("Starting complete graph RAG processing pipeline")
        start_time = time.time()

        # Step 1: Process PDFs
        logger.info("Step 1: Processing PDF files")
        chunks, source_metadata = process_pdfs(pdf_files)
        if not chunks:
            logger.error("No valid chunks extracted from PDFs")
            return "Error: No valid content extracted from PDF files"
        
        # Step 2: Connect to Neo4j
        logger.info("Step 2: Connecting to Neo4j and setting up schema")
        graph = connect_to_neo4j(neo4j_uri, neo4j_username, neo4j_password)
        setup_neo4j_schema(graph)
        
        # Step 3: Initialize embedding model
        logger.info("Step 3: Initializing embedding model")
        embedding_function = get_embeddings()
        
        # Step 4: Create knowledge graph
        logger.info("Step 4: Building knowledge graph from document chunks")
        create_knowledge_graph(graph, chunks, source_metadata, api_key)
        
        # Step 5: Setup vector index
        logger.info("Step 5: Creating vector index for semantic search")
        vector_store = setup_vector_index(chunks, embedding_function, neo4j_uri, neo4j_username, neo4j_password)
        
        # Step 6: Generate graph statistics
        logger.info("Step 6: Generating graph statistics")
        doc_count = graph.query("MATCH (d:Document) RETURN count(d) as count")[0]['count']
        chunk_count = graph.query("MATCH (c:Chunk) RETURN count(c) as count")[0]['count']
        entity_count = graph.query("MATCH (e:Entity) RETURN count(e) as count")[0]['count']
        
        # Global retriever setup for the app
        global_data = {
            "graph": graph,
            "vector_store": vector_store,
            "neo4j_uri": neo4j_uri,
            "neo4j_username": neo4j_username,
            "neo4j_password": neo4j_password,
            "doc_count": doc_count,
            "chunk_count": chunk_count,
            "entity_count": entity_count
        }
        
        elapsed_time = time.time() - start_time
        logger.info(f"Graph RAG pipeline completed in {elapsed_time:.2f} seconds")
        return global_data
        
    except Exception as e:
        logger.error(f"Critical error in processing pipeline: {str(e)}", exc_info=True)
        return f"Critical error: {str(e)}"

def ask_graph_rag(question, api_key, global_data):
    """Full QA pipeline using graph RAG approach"""
    logger.info(f"Processing question: {question}")
    start_time = time.time()
    
    try:
        if not global_data or "graph" not in global_data:
            logger.error("Error: Process PDFs and build graph first")
            return "Error: Process PDFs and build graph first"
            
        if not api_key:
            logger.error("Error: Missing OpenRouter API key")
            return "Error: Please enter your OpenRouter API key"
            
        # Get combined context from vector and graph
        logger.info("Getting context from vector store and knowledge graph")
        contexts = get_query_context(
            question, 
            global_data["vector_store"], 
            global_data["graph"]
        )
        
        # Get answer using context
        logger.info("Generating answer based on context")
        answer = answer_question(
            question, 
            api_key, 
            contexts, 
            global_data["neo4j_uri"], 
            global_data["neo4j_username"], 
            global_data["neo4j_password"]
        )
        
        # Log the contexts and answer for debugging
        logger.debug(f"Contexts used for answer: {json.dumps([{k: c[k] for k in ['source', 'chunk_id']} for c in contexts])}")
        
        # Format answer for display
        formatted_answer = f"""Answer: {answer}

Processing Stats:
- Query time: {time.time() - start_time:.2f} seconds
- Contexts used: {len(contexts)}
- Context sources: {', '.join(set([c['source'] for c in contexts]))}
"""
        logger.info(f"Question answered in {time.time() - start_time:.2f} seconds")
        return formatted_answer
        
    except Exception as e:
        logger.error(f"Error in QA pipeline: {str(e)}", exc_info=True)
        return f"Error processing question: {str(e)}"

# =============================
# Debug and Visualization Functions
# =============================
def debug_knowledge_graph(global_data):
    """Generate debug information about the knowledge graph"""
    logger.info("Generating knowledge graph debug information")
    
    try:
        if not global_data or "graph" not in global_data:
            return "Error: Process PDFs and build graph first"
            
        graph = global_data["graph"]
        
        # Document stats
        doc_query = """
        MATCH (d:Document)
        RETURN d.id as document, count{(d)-[:CONTAINS]->(:Chunk)} as chunks
        ORDER BY chunks DESC
        """
        doc_stats = graph.query(doc_query)
        
        # Entity stats
        entity_query = """
        MATCH (e:Entity)
        WITH e.type as entity_type, count(e) as count
        ORDER BY count DESC
        RETURN entity_type, count
        LIMIT 10
        """
        entity_stats = graph.query(entity_query)
        
        # Relationship stats
        rel_query = """
        MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
        RETURN e1.name as source, e2.name as target, r.weight as weight
        ORDER BY r.weight DESC
        LIMIT 10
        """
        rel_stats = graph.query(rel_query)
        
        # Format debug info
        debug_info = "Knowledge Graph Debug Information\n"
        debug_info += "================================\n\n"
        
        debug_info += f"Total Documents: {global_data['doc_count']}\n"
        debug_info += f"Total Chunks: {global_data['chunk_count']}\n"
        debug_info += f"Total Entities: {global_data['entity_count']}\n\n"
        
        debug_info += "Documents and Chunks:\n"
        debug_info += "-----------------------\n"
        for doc in doc_stats:
            debug_info += f"Document: {doc['document']} - Chunks: {doc['chunks']}\n"
        
        debug_info += "\nTop Entity Types:\n"
        debug_info += "-----------------------\n"
        for entity in entity_stats:
            debug_info += f"Type: {entity['entity_type']} - Count: {entity['count']}\n"
        
        debug_info += "\nTop Entity Relationships:\n"
        debug_info += "-----------------------\n"
        for rel in rel_stats:
            debug_info += f"{rel['source']} -> {rel['target']} (Weight: {rel['weight']})\n"
            
        logger.info("Debug information generated successfully")
        return debug_info
        
    except Exception as e:
        logger.error(f"Error generating debug info: {str(e)}", exc_info=True)
        return f"Error generating debug info: {str(e)}"

def view_query_execution(question, global_data):
    """View the step-by-step query execution process"""
    logger.info(f"Analyzing query execution for: {question}")
    
    try:
        if not global_data or "graph" not in global_data:
            return "Error: Process PDFs and build graph first"
            
        # Start with entity extraction
        entities = extract_entities_from_question(question)
        
        # Log the steps
        execution_log = "Query Execution Analysis\n"
        execution_log += "=======================\n\n"
        
        execution_log += f"Question: {question}\n\n"
        
        execution_log += "Step 1: Entity Extraction\n"
        execution_log += "-----------------------\n"
        execution_log += f"Entities identified: {', '.join(entities) if entities else 'None'}\n\n"
        
        execution_log += "Step 2: Vector Search\n"
        execution_log += "-----------------------\n"
        vector_results = global_data["vector_store"].similarity_search(question, k=3)
        for i, doc in enumerate(vector_results):
            execution_log += f"Result {i+1}: {doc.metadata.get('chunk_id', 'unknown')}\n"
            execution_log += f"Similarity: Using embedding vectors\n"
            execution_log += f"Content (truncated): {doc.page_content[:150]}...\n\n"
        
        execution_log += "Step 3: Graph Traversal\n"
        execution_log += "-----------------------\n"
        if entities:
            for entity in entities:
                execution_log += f"Searching for entity: {entity}\n"
                query = """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($entity_name)
                RETURN c.id as chunk_id, e.name as entity_name
                LIMIT 3
                """
                results = global_data["graph"].query(query, params={"entity_name": entity})
                for result in results:
                    execution_log += f"Found chunk: {result['chunk_id']} mentioning entity: {result['entity_name']}\n"
                
                execution_log += "\nChecking for related entities:\n"
                query = """
                MATCH (e1:Entity)-[r:RELATED_TO]->(e2:Entity)
                WHERE toLower(e1.name) CONTAINS toLower($entity_name)
                RETURN e1.name as source, e2.name as related, r.weight as strength
                ORDER BY r.weight DESC
                LIMIT 3
                """
                results = global_data["graph"].query(query, params={"entity_name": entity})
                for result in results:
                    execution_log += f"Related: {result['source']} -> {result['related']} (strength: {result['strength']})\n"
                
                execution_log += "\n"
        else:
            execution_log += "No entities found for graph traversal\n\n"
        
        execution_log += "Step 4: Context Combination\n"
        execution_log += "-----------------------\n"
        execution_log += "Final context will combine vector results and graph traversal results\n"
        
        logger.info("Query execution analysis completed")
        return execution_log
        
    except Exception as e:
        logger.error(f"Error analyzing query execution: {str(e)}", exc_info=True)
        return f"Error analyzing query execution: {str(e)}"

# Global data store
global_data = None

# =============================
# Gradio Interface
# =============================
def create_gradio_interface():
    """Create Gradio interface for the Graph RAG application"""
    logger.info("Creating Gradio interface")
    
    with gr.Blocks(title="Graph RAG Assistant") as demo:
        gr.Markdown("# Graph RAG Question Answering System")
        gr.Markdown("This system uses a knowledge graph and vector embeddings to answer questions about your documents.")
        
        with gr.Tab("Setup"):
            gr.Markdown("## Step 1: Configure Neo4j Connection")
            with gr.Row():
                neo4j_uri = gr.Textbox(
                    label="Neo4j URI", 
                    placeholder="bolt://localhost:7687",
                    value="bolt://localhost:7687"
                )
                neo4j_username = gr.Textbox(
                    label="Neo4j Username",
                    placeholder="neo4j",
                    value="neo4j"
                )
                neo4j_password = gr.Textbox(
                    label="Neo4j Password",
                    type="password",
                    placeholder="Enter your Neo4j password"
                )
            
            gr.Markdown("## Step 2: Upload PDF Documents")
            with gr.Row():
                pdf_upload = gr.Files(
                    label="PDF Documents",
                    file_types=[".pdf"],
                    file_count="multiple"
                )
                api_key_setup = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    placeholder="Enter your OpenRouter API key here"
                )
                
            process_btn = gr.Button("Process Documents and Build Graph", variant="primary")
            status = gr.Textbox(label="Processing Status", lines=10)
            
        with gr.Tab("Ask Questions"):
            gr.Markdown("## Ask Questions About Your Documents")
            with gr.Row():
                api_key = gr.Textbox(
                    label="OpenRouter API Key",
                    type="password",
                    placeholder="Enter your OpenRouter API key here"
                )
            question = gr.Textbox(
                label="Your Question",
                placeholder="What would you like to know from the documents?"
            )
            ask_btn = gr.Button("Get Answer", variant="primary")
            answer = gr.Textbox(label="Answer", interactive=False, lines=10)

        with gr.Tab("Debug & Visualization"):
            gr.Markdown("## Debug and Visualize Knowledge Graph")
            with gr.Row():
                debug_btn = gr.Button("Generate Debug Info", variant="secondary")
                viz_btn = gr.Button("Visualize Graph", variant="secondary")
            
            with gr.Accordion("Advanced Debug Options"):
                debug_question = gr.Textbox(
                    label="Debug Query Process",
                    placeholder="Enter a question to see how it would be processed"
                )
                debug_query_btn = gr.Button("Analyze Query Execution", variant="secondary")
            
            debug_output = gr.HTML(label="Debug Output")
            
        with gr.Tab("Logs"):
            gr.Markdown("## System Logs")
            refresh_log_btn = gr.Button("Refresh Logs", variant="secondary")
            log_output = gr.Textbox(label="Log Output", lines=20)
        
        # Define functions to handle button clicks
        def process_and_build(neo4j_uri_value, neo4j_username_value, neo4j_password_value, pdf_files, api_key_value):
          global global_data
          if not neo4j_uri_value or not neo4j_username_value or not neo4j_password_value:
              return "Error: Please provide Neo4j connection details"
          if not pdf_files:
              return "Error: Please upload at least one PDF file"
          if not api_key_value:
              return "Error: Please provide your OpenRouter API key"
          
          result = process_files_and_build_graph(
                    pdf_files,
                    neo4j_uri_value,
                    neo4j_username_value,
                    neo4j_password_value,
                    api_key_value
                    )
          if isinstance(result, dict):
              global_data = result
              return f"Processing complete! Added {result['doc_count']} documents with {result['chunk_count']} chunks and {result['entity_count']} entities to the graph."
          else:
              return result
        
        def ask_question(question_value, api_key_value):
            global global_data
            if not global_data:
                return "Error: Please process documents first"
            return ask_graph_rag(question_value, api_key_value, global_data)
        
        def get_debug_info():
            global global_data
            if not global_data:
                return "Error: Please process documents first"
            debug_info = debug_knowledge_graph(global_data)
            return f"<pre>{debug_info}</pre>"
        
        def get_visualization():
            global global_data
            if not global_data:
                return "Error: Please process documents first"
            return visualize_graph(
                global_data["neo4j_uri"], 
                global_data["neo4j_username"], 
                global_data["neo4j_password"]
            )
        
        def analyze_query(debug_question_value):
            global global_data
            if not global_data:
                return "Error: Please process documents first"
            if not debug_question_value:
                return "Error: Please enter a question to analyze"
            execution_log = view_query_execution(debug_question_value, global_data)
            return f"<pre>{execution_log}</pre>"
        
        def refresh_logs():
            try:
                with open('graph_rag.log', 'r') as f:
                    # Get the last 50 lines of the log file
                    lines = f.readlines()
                    last_lines = lines[-50:] if len(lines) > 50 else lines
                    return "".join(last_lines)
            except Exception as e:
                return f"Error reading log file: {str(e)}"
        
        # Connect UI elements to functions
        process_btn.click(process_and_build, inputs=[neo4j_uri, neo4j_username, neo4j_password, pdf_upload, api_key_setup], outputs=status)
        ask_btn.click(ask_question, inputs=[question, api_key], outputs=answer)
        debug_btn.click(get_debug_info, inputs=[], outputs=debug_output)
        viz_btn.click(get_visualization, inputs=[], outputs=debug_output)
        debug_query_btn.click(analyze_query, inputs=[debug_question], outputs=debug_output)
        refresh_log_btn.click(refresh_logs, inputs=[], outputs=log_output)

        
        # Add listeners to update API key across tabs
        api_key_setup.change(
            lambda x: x, 
            inputs=[api_key_setup], 
            outputs=[api_key]
        )
        api_key.change(
            lambda x: x, 
            inputs=[api_key], 
            outputs=[api_key_setup]
        )
        
    return demo

# =============================
# Main Function
# =============================
def main():
    """Main function to run the application"""
    logger.info("Starting Graph RAG application")
    
    # Display header
    print("="*50)
    print("Graph RAG System for PDF Analysis")
    print("="*50)
    print("1. Install dependencies if not already installed")
    print("2. Configure Neo4j connection")
    print("3. Upload PDFs to process")
    print("4. Ask questions about your documents")
    print("="*50)
    
    # Create and launch the Gradio interface
    demo = create_gradio_interface()
    demo.launch(share=True)

if __name__ == "__main__":
    main()