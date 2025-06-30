# Phase 3: Data Integration - Complete Study Notes
*Weeks 5-6: Document Processing, Vector Databases, and RAG Systems*

---

## 3.1 Document Processing

### Document Loaders

#### PDF Processing
- **PyPDF2/PyPDF4**: Basic PDF text extraction
- **pdfplumber**: Advanced PDF parsing with table support
- **pymupdf (fitz)**: High-performance PDF processing
- **Unstructured**: Handles complex layouts and mixed content

```python
# Example: LangChain PDF loader
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
pages = loader.load_and_split()
```

**Challenges with PDFs:**
- Scanned documents (requires OCR)
- Complex layouts and tables
- Embedded images and metadata
- Password-protected files

#### Word Document Processing
- **python-docx**: Native Word document parsing
- **mammoth**: Convert .docx to HTML/markdown
- **pandoc**: Universal document converter

```python
# Example: Word document loader
from langchain.document_loaders import Docx2txtLoader
loader = Docx2txtLoader("document.docx")
documents = loader.load()
```

#### CSV Processing
- **pandas**: Data manipulation and analysis
- **csv module**: Built-in Python CSV handling
- **Structured vs. unstructured**: CSV as data vs. text content

```python
# Example: CSV as documents
from langchain.document_loaders import CSVLoader
loader = CSVLoader("data.csv", 
                   csv_args={'delimiter': ',',
                            'quotechar': '"'})
documents = loader.load()
```

#### Web Scraping
- **Beautiful Soup**: HTML parsing and extraction
- **Scrapy**: Large-scale web scraping framework
- **Selenium**: Dynamic content and JavaScript rendering
- **Requests-HTML**: Simple async web scraping

```python
# Example: Web page loader
from langchain.document_loaders import WebBaseLoader
loader = WebBaseLoader("https://example.com")
documents = loader.load()
```

**Web Scraping Considerations:**
- Respect robots.txt
- Rate limiting and politeness
- Dynamic content handling
- Legal and ethical considerations

### Text Splitters

#### Chunking Strategies

**1. Fixed-Size Chunking**
- Split by character count or token count
- Simple but may break semantic meaning
- Good for consistent processing

```python
from langchain.text_splitter import CharacterTextSplitter
splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n\n"
)
```

**2. Semantic Chunking**
- Split by paragraphs, sentences, or topics
- Preserves meaning and context
- More complex but better quality

**3. Recursive Chunking**
- Hierarchical splitting approach
- Try different separators in order
- Falls back to character-based splitting

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)
```

**4. Document-Aware Chunking**
- Consider document structure (headers, sections)
- Maintain document hierarchy
- Preserve formatting and metadata

#### Chunking Parameters
- **Chunk Size**: Balance between context and specificity
- **Overlap**: Prevents loss of context at boundaries
- **Separators**: Natural break points in text
- **Length Function**: Characters, tokens, or words

#### Best Practices
- Test different chunk sizes for your use case
- Consider your embedding model's context window
- Preserve important structural elements
- Monitor chunk quality and coherence

### Document Transformers

#### Text Preprocessing
- **Cleaning**: Remove unwanted characters, formatting
- **Normalization**: Standardize text format
- **Language detection**: Handle multilingual content
- **Encoding**: Ensure proper character encoding

#### Content Enhancement
- **Metadata extraction**: Titles, authors, dates
- **Section identification**: Headers, chapters, topics
- **Link resolution**: Handle references and citations
- **Image/table extraction**: Process non-text elements

#### Custom Transformers
```python
from langchain.document_transformers import BaseDocumentTransformer

class CustomTransformer(BaseDocumentTransformer):
    def transform_documents(self, documents):
        # Custom transformation logic
        for doc in documents:
            doc.page_content = self.process_text(doc.page_content)
        return documents
```

---

## 3.2 Vector Databases and Embeddings

### Understanding Embeddings

#### What are Embeddings?
- **Vector representations**: Numerical representations of text
- **Semantic meaning**: Similar concepts have similar vectors
- **High-dimensional space**: Typically 384-4096 dimensions
- **Dense vectors**: Most values are non-zero

#### Types of Embeddings
**1. Word Embeddings**
- Word2Vec, GloVe, FastText
- Individual word representations
- Limited context understanding

**2. Sentence/Document Embeddings**
- BERT, RoBERTa, SentenceTransformers
- Context-aware representations
- Better for document retrieval

**3. Specialized Embeddings**
- Code embeddings (CodeBERT)
- Multilingual embeddings
- Domain-specific models

#### Popular Embedding Models
- **OpenAI**: text-embedding-ada-002, text-embedding-3-small/large
- **Sentence Transformers**: all-MiniLM-L6-v2, all-mpnet-base-v2
- **Cohere**: embed-english-v3.0
- **Google**: Universal Sentence Encoder

```python
# Example: Using OpenAI embeddings
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

### Vector Stores

#### Chroma
- **Open-source**: Free and lightweight
- **Python-native**: Easy integration
- **Local storage**: Good for development
- **Metadata filtering**: Query by document properties

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

#### Pinecone
- **Cloud-based**: Managed service
- **Scalable**: Handle millions of vectors
- **Fast queries**: Optimized for similarity search
- **Metadata support**: Rich filtering capabilities

```python
import pinecone
from langchain.vectorstores import Pinecone

pinecone.init(api_key="your-api-key", environment="your-env")
vectorstore = Pinecone.from_documents(
    documents=docs,
    embedding=embeddings,
    index_name="your-index"
)
```

#### FAISS (Facebook AI Similarity Search)
- **High performance**: Optimized for speed
- **Memory efficient**: Good for large datasets
- **No cloud dependency**: Runs locally
- **Multiple indices**: Different optimization strategies

```python
from langchain.vectorstores import FAISS

vectorstore = FAISS.from_documents(
    documents=docs,
    embedding=embeddings
)
# Save and load
vectorstore.save_local("faiss_index")
```

#### Comparison of Vector Stores

| Feature | Chroma | Pinecone | FAISS |
|---------|---------|----------|--------|
| **Deployment** | Local/Cloud | Cloud | Local |
| **Scalability** | Medium | High | High |
| **Cost** | Free | Paid | Free |
| **Ease of Use** | High | High | Medium |
| **Performance** | Good | Excellent | Excellent |
| **Metadata** | Yes | Yes | Limited |

### Similarity Search

#### Distance Metrics
**1. Cosine Similarity**
- Measures angle between vectors
- Good for normalized embeddings
- Range: -1 to 1 (higher is more similar)

**2. Euclidean Distance**
- Measures straight-line distance
- Sensitive to vector magnitude
- Range: 0 to ∞ (lower is more similar)

**3. Dot Product**
- Measures alignment and magnitude
- Fast to compute
- Good for sparse vectors

#### Search Strategies
**1. k-NN Search**
- Find k nearest neighbors
- Simple but effective
- May miss diverse results

**2. Threshold Search**
- Return all results above similarity threshold
- Variable result count
- Good for quality control

**3. MMR (Maximal Marginal Relevance)**
- Balance relevance and diversity
- Reduces redundant results
- Better for varied information needs

```python
# Example: Different search methods
# Basic similarity search
results = vectorstore.similarity_search("query", k=5)

# Search with scores
results = vectorstore.similarity_search_with_score("query", k=5)

# MMR search
results = vectorstore.max_marginal_relevance_search("query", k=5)
```

### Retrieval Strategies

#### Dense Retrieval
- Uses dense vector embeddings
- Good for semantic similarity
- Requires pre-computed embeddings

#### Sparse Retrieval
- Uses traditional keyword matching (BM25, TF-IDF)
- Good for exact matches
- Fast and interpretable

#### Hybrid Retrieval
- Combines dense and sparse methods
- Best of both approaches
- More complex to implement

```python
# Example: Hybrid retrieval
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# Dense retriever
dense_retriever = vectorstore.as_retriever()

# Sparse retriever
sparse_retriever = BM25Retriever.from_documents(documents)

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, sparse_retriever],
    weights=[0.7, 0.3]
)
```

---

## 3.3 Retrieval-Augmented Generation (RAG)

### Basic RAG

#### RAG Components
1. **Retriever**: Finds relevant documents
2. **Generator**: Creates responses using retrieved context
3. **Prompt Template**: Combines query and context

#### Simple RAG Pipeline
```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# Create RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Query the system
result = qa_chain({"query": "What is the main topic?"})
```

#### Chain Types
**1. Stuff Chain**
- Puts all retrieved documents into prompt
- Simple but limited by token limits
- Good for small document sets

**2. Map-Reduce Chain**
- Processes documents separately, then combines
- Handles large document sets
- More complex but scalable

**3. Refine Chain**
- Iteratively refines answer with each document
- Good for building comprehensive answers
- Sequential processing

**4. Map-Rerank Chain**
- Scores each document's relevance
- Returns best single answer
- Good for factual questions

### Advanced RAG

#### Multi-Document Retrieval
- **Cross-document reasoning**: Connect information across sources
- **Document ranking**: Prioritize most relevant sources
- **Conflict resolution**: Handle contradictory information

#### Query Enhancement
**1. Query Expansion**
- Add related terms to improve retrieval
- Use synonyms and related concepts
- Generate multiple query variations

**2. Query Decomposition**
- Break complex queries into sub-questions
- Answer each part separately
- Combine results into final answer

**3. Query Routing**
- Direct queries to appropriate knowledge bases
- Different strategies for different query types
- Improve efficiency and accuracy

#### Context Management
**1. Context Compression**
- Summarize retrieved documents
- Remove irrelevant information
- Fit more information in context window

**2. Context Ranking**
- Reorder retrieved documents by relevance
- Use cross-encoder models for ranking
- Improve answer quality

```python
# Example: Advanced RAG with compression
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

# Create compressor
compressor = LLMChainExtractor.from_llm(llm)

# Create compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=vectorstore.as_retriever()
)
```

### Retrieval Chains

#### RetrievalQA Chain
- Basic question-answering over documents
- Single-turn conversation
- Good for factual queries

```python
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)
```

#### ConversationalRetrievalChain
- Multi-turn conversations with memory
- Maintains chat history
- Good for interactive applications

```python
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

conv_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
```

#### Custom Chains
- Build application-specific logic
- Combine multiple retrieval strategies
- Add preprocessing and postprocessing

### RAG Evaluation

#### Retrieval Metrics
**1. Precision@k**
- Fraction of retrieved documents that are relevant
- Measures retrieval accuracy

**2. Recall@k**
- Fraction of relevant documents that are retrieved
- Measures retrieval completeness

**3. Mean Reciprocal Rank (MRR)**
- Average of reciprocal ranks of first relevant result
- Measures ranking quality

#### Generation Metrics
**1. Faithfulness**
- How well the answer reflects the source documents
- Prevents hallucination

**2. Answer Relevance**
- How well the answer addresses the question
- Measures response quality

**3. Context Relevance**
- How relevant the retrieved context is to the question
- Measures retrieval quality

#### Evaluation Frameworks
```python
# Example: RAGAS evaluation
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy

# Prepare evaluation data
evaluation_data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Run evaluation
results = evaluate(
    evaluation_data,
    metrics=[faithfulness, answer_relevancy]
)
```

#### A/B Testing
- Compare different RAG configurations
- Test various embedding models
- Optimize chunk sizes and retrieval parameters

---

## 3.4 Projects

### Project 1: Document Q&A System

#### Objectives
- Build end-to-end document processing pipeline
- Implement basic RAG functionality
- Create simple user interface

#### Implementation Steps
1. **Document Ingestion**
   - Support multiple file formats
   - Extract and clean text
   - Create document metadata

2. **Vector Store Setup**
   - Choose appropriate embedding model
   - Configure vector database
   - Index documents efficiently

3. **Query Interface**
   - Design user-friendly interface
   - Implement query processing
   - Display results with sources

#### Technical Requirements
```python
# Example structure
class DocumentQASystem:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vectorstore = Chroma(embedding_function=self.embeddings)
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=OpenAI(),
            retriever=self.vectorstore.as_retriever()
        )
    
    def add_documents(self, file_paths):
        # Document processing logic
        pass
    
    def query(self, question):
        # Query processing logic
        pass
```

### Project 2: Knowledge Base Chatbot

#### Objectives
- Create conversational interface
- Maintain conversation history
- Handle multi-turn interactions

#### Advanced Features
- **Context awareness**: Remember previous questions
- **Follow-up handling**: Process clarifying questions
- **Source attribution**: Show where answers come from
- **Conversation export**: Save chat histories

#### Implementation Considerations
```python
# Example: Conversational system
class KnowledgeBaseChatbot:
    def __init__(self):
        self.memory = ConversationBufferWindowMemory(k=5)
        self.conv_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory
        )
    
    def chat(self, message):
        response = self.conv_chain({"question": message})
        return response
```

### Project 3: Research Assistant Tool

#### Objectives
- Support complex research workflows
- Handle multiple document types
- Provide comprehensive analysis

#### Advanced Capabilities
- **Multi-document synthesis**: Combine information across sources
- **Citation management**: Track and format references
- **Research planning**: Break down complex research questions
- **Export functionality**: Generate reports and summaries

#### Architecture Suggestions
1. **Modular Design**
   - Separate components for different functions
   - Easy to extend and maintain

2. **Scalable Storage**
   - Handle large document collections
   - Efficient indexing and retrieval

3. **User Experience**
   - Intuitive interface for researchers
   - Powerful query capabilities
   - Export options for different formats

---

## Best Practices and Tips

### Document Processing
- Always validate input documents
- Handle encoding issues gracefully
- Preserve important metadata
- Monitor processing performance

### Vector Store Management
- Regular index maintenance
- Monitor storage usage
- Backup important indices
- Test retrieval quality regularly

### RAG Optimization
- Experiment with different chunk sizes
- Test various embedding models
- Optimize prompt templates
- Monitor response quality

### Production Considerations
- Implement proper error handling
- Add logging and monitoring
- Consider rate limiting
- Plan for scaling

### Common Pitfalls
- **Chunk size too large**: May exceed context limits
- **Chunk size too small**: May lose important context
- **Poor document quality**: Garbage in, garbage out
- **Inadequate evaluation**: Deploy without proper testing
- **Ignoring edge cases**: Unusual queries or documents

---

## Additional Resources

### Tools and Libraries
- **LangChain**: Comprehensive RAG framework
- **LlamaIndex**: Alternative RAG library
- **Haystack**: Open-source NLP framework
- **txtai**: Semantic search and workflows

### Embedding Models
- **Hugging Face**: Wide variety of models
- **OpenAI**: High-quality commercial embeddings
- **Cohere**: Specialized embedding models
- **Sentence Transformers**: Open-source options

### Vector Databases
- **Weaviate**: GraphQL-based vector database
- **Qdrant**: High-performance vector search
- **Milvus**: Scalable vector database
- **pgvector**: PostgreSQL extension for vectors

### Evaluation Tools
- **RAGAS**: RAG assessment framework
- **TruLens**: LLM application evaluation
- **Phoenix**: ML observability platform
- **LangSmith**: LangChain debugging and evaluation
