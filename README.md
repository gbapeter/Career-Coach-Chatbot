# Data Science & Tech Career Advisor
### AI/ML – Group 4 | Technical Report

> **A RAG-Based AI Career Mentorship Chatbot**
> *April 2026*

**By:**
Peter Gba · Amina Isa · Hauwa'u Al-Bashir · Nusaiba Abdulnasir

---

## Abstract

This report presents the design, development, and evaluation of a Retrieval-Augmented Generation (RAG) chatbot built to serve as an AI career mentor for aspiring Data Scientists and technology professionals. The system combines a curated library of industry documents (including career roadmaps, interview guides, skill frameworks, and salary reports) with a large language model to deliver grounded, source-attributed career advice. Unlike general-purpose AI assistants, this chatbot is strictly constrained to respond only from verified career resources, thereby eliminating fabricated information. The application features a professional mentor persona engineered through prompt design, interactive example prompts for user onboarding, and full source attribution displayed beneath every response. This report covers the end-to-end system architecture, the data ingestion pipeline, retrieval mechanisms, prompt engineering strategy, interface design, and a complete technical evaluation.

🔗 **Live App:** [https://career-coach-chatbot-tv.streamlit.app/](https://career-coach-chatbot-tv.streamlit.app/)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Architecture](#2-system-architecture)
3. [Data Curation and Knowledge Base](#3-data-curation-and-knowledge-base)
4. [Prompt Engineering and Persona Design](#4-prompt-engineering-and-persona-design)
5. [Application Features and Interface](#5-application-features-and-interface)
6. [Evaluation and Limitations](#6-evaluation-and-limitations)
7. [Deployment](#7-deployment)
8. [Group Contributions](#8-group-contributions)
9. [Conclusion](#9-conclusion)

---

## 1. Introduction

### 1.1 Background and Motivation

The rapid expansion of the data science and technology industry has created an urgent demand for accessible, reliable career guidance. Aspiring professionals frequently navigate a fragmented landscape of online content, much of which is outdated, commercially biased, or factually unreliable. Traditional AI chatbots that rely solely on large language model (LLM) training data risk compounding this problem by generating hallucinated or obsolete information such as incorrect salary figures, deprecated tool recommendations, or non-existent certifications.

Retrieval-Augmented Generation (RAG) directly addresses this limitation by anchoring AI responses to specific, curated documents rather than relying exclusively on parametric knowledge encoded during pretraining. This project applies the RAG paradigm to build a focused, trustworthy career advisor that operates entirely within a verified knowledge base assembled from reputable industry sources.

### 1.2 Project Aim and Objectives

**Aim**

To develop a RAG-based AI chatbot that provides reliable, source-attributed career guidance to aspiring Data Scientists and technology professionals, using curated industry documents and prompt engineering.

**Objectives**

- Curate a knowledge base of at least 30 authoritative industry documents by the project submission date.
- Build a document ingestion pipeline that chunks, embeds, and stores all sources in ChromaDB without errors.
- Engineer a mentor persona entirely through prompt design such that the chatbot maintains tone and declines out-of-scope questions in at least 90% of test cases.
- Develop a functional Streamlit web application with a chat interface, example prompt buttons, and source attribution, deployed via a public URL before the submission deadline.
- Evaluate system performance across 25 test questions, achieving strong ratings in groundedness, relevance, and attribution accuracy.

### 1.3 Constraints

The project operates under one critical constraint: no fine-tuning of the base language model is permitted. All domain adaptation is achieved exclusively through document retrieval and prompt engineering. The system is scoped to career guidance topics within data science and technology and does not handle queries outside this domain.

---

## 2. System Architecture

### 2.1 Architectural Overview

The system is composed of two distinct, decoupled pipelines. The **offline ingestion pipeline** runs once (or whenever the knowledge base is refreshed) and constructs the searchable vector database. The **online query pipeline** serves user requests in real time, retrieving relevant context and generating grounded responses on demand.

### 2.2 Offline Ingestion Pipeline

The ingestion pipeline (`ingest.py`) transforms raw career documents into a searchable vector store through four sequential stages:

1. **Document Collection:** PDF career guides are downloaded from curated URLs using the `requests` library and parsed page-by-page with `PyPDFLoader`. Web-based resources are fetched and cleaned using `BeautifulSoup`, which strips navigation menus, scripts, advertisements, and footer noise, retaining only substantive article content.

2. **Text Chunking:** All extracted text is segmented into 800-character chunks with a 150-character overlap using LangChain's `RecursiveCharacterTextSplitter`. The splitter prioritises natural language boundaries like paragraphs, sentences, and clauses before resorting to word or character splits, preserving semantic coherence within each chunk.

3. **Embedding:** Each chunk is vectorised using HuggingFace sentence-transformers model (`all-MiniLM-L6-v2`), which maps text to a 384-dimensional dense vector space where semantically similar passages are geometrically proximate.

4. **Vector Storage:** Chunk vectors and associated metadata (including the source URL and document type) persist in a local ChromaDB instance under the collection name `career_advisor_2026`. The vector database is built once locally, compressed into a ZIP file, and committed to the project repository. On deployment, the application automatically extracts the pre-built database, eliminating the need to re-run the ingestion pipeline in production.

### 2.3 Online Query Pipeline

When a user submits a query, the following real-time pipeline executes:

1. The user's query is embedded using the same HuggingFace `all-MiniLM-L6-v2` model used during ingestion, ensuring vector space consistency.
2. ChromaDB performs a cosine similarity search across all stored chunk vectors, returning the top-6 most semantically relevant passages (`k=6`).
3. The retrieved chunks are injected into the LLM's context window alongside the system prompt and the user's question.
4. The language model generates a response grounded exclusively in the provided context. The full response is returned once completely generated.
5. The source URLs stored in each retrieved chunk's metadata are deduplicated and displayed beneath the answer as clickable citations.

### 2.4 Technology Stack

| Component | Technology | Role in System |
|---|---|---|
| Document Loading | PyPDFLoader, BeautifulSoup4 | Extract clean text from PDFs and web pages |
| Text Splitting | RecursiveCharacterTextSplitter | Chunk documents into semantically coherent segments |
| Embedding Model | all-MiniLM-L6-v2 (HuggingFace) | Convert text chunks and queries to 384-dim dense vectors |
| Vector Store | ChromaDB (local persistent) | Store, index, and search embedded chunks via cosine similarity |
| Language Model | Gemini 2.5 Flash (Google) | Generate grounded, mentor-toned responses |
| Orchestration | LangChain | Connect and standardise all pipeline components |
| UI Framework | Streamlit 1.33.0 | Deliver the interactive user-facing chat interface |
| HTTP Client | requests 2.31.0 | Download PDFs and web pages for ingestion |
| HTML Parser | BeautifulSoup4 4.12.3 | Clean and extract readable text from HTML pages |

---

## 3. Data Curation and Knowledge Base

### 3.1 Source Selection Criteria

Documents were selected according to four criteria: **authority** (published by recognised industry bodies, established practitioners, or peer-reviewed sources), **recency** (published or substantively updated within the past three years), **relevance** (directly applicable to data science or tech career progression), and **accessibility** (freely and legally available for programmatic ingestion).

### 3.2 Document Categories and Sources

The knowledge base comprises 6 PDF documents and 25 web-scraped URLs, spanning five content categories:

| Category | Key Sources | Career Value |
|---|---|---|
| Career Roadmaps | roadmap.sh/ai-data-scientist, GeeksforGeeks, AnalyticsVidhya, upGrad, Scaler | Structured learning paths from beginner to senior |
| Interview Guides | 365 Data Science Interview Guide, SimpliLearn Interview Guide | Prepares users for real technical and behavioural hiring processes |
| Industry Reports | WEF Future of Jobs 2025, Stack Overflow Survey 2024, JetBrains State of Data Science, LinkedIn Skills on the Rise 2025 | Evidence-based skill demand and salary benchmarks |
| Portfolio Guides | 365 Data Science, DataQuest (x2), BrainStation, freeCodeCamp, Towards Data Science | Actionable advice on building credibility and project showcasing |
| Job Market & Salary | InterviewQuery, Syracuse iSchool, Tredence, RefonteLeaning, TCS iON, Gulf News | Helps users prioritise skills and set salary expectations |
| PDF Reports | Code With Mosh Data Science Roadmap, WEF Future of Jobs 2025, WEF New Economy Skills 2025, WEF Human Advantage 2026, LinkedIn Job Skills 2026, IntaPeople Tech Guide 2026 | Authoritative long-form reference documents |

### 3.3 Chunking Strategy and Rationale

After experimentation, a chunk size of **800 characters** with **150-character overlap** was selected. The splitter uses a priority hierarchy of separators: double newlines (paragraph breaks), single newlines, sentence-ending punctuation (`'. '`, `'!'`, `'?'`), spaces, falling back to raw characters only as a last resort. This preserves natural language coherence within each chunk.

Chunks smaller than 500 characters improved retrieval precision but often lacked sufficient context for the LLM to produce coherent answers. Chunks larger than 1,200 characters improved context but reduced retrieval specificity, causing irrelevant passages to be included. The 150-character overlap ensures that sentences bisected at a chunk boundary retain their full meaning in at least one of the two adjacent chunks, preventing information loss at boundaries.

---

## 4. Prompt Engineering and Persona Design

### 4.1 System Prompt Architecture

The chatbot's behaviour, tone, and constraints are governed entirely by a carefully engineered system prompt. Because no fine-tuning is used, the system prompt is the sole mechanism for shaping the model's identity and output style. The deployed system prompt is:

```
You are CareerAI, a warm, experienced, professional, and encouraging Data
Science and Tech Career Advisor.

Answer ONLY based on the retrieved documents.
If the information is not in the context, say: "I don't have that specific
information in my knowledge base."
Always be encouraging and give actionable advice.
```

### 4.2 Mentor Persona Design Principles

- **Professional yet approachable:** formal enough to convey expertise, warm enough to reduce anxiety in users seeking career advice
- **Honest about scope:** the bot explicitly states when a question falls outside its knowledge base rather than speculating
- **Action-oriented:** every response concludes with at least one concrete next step the user can take
- **Grounded:** all claims are tied to retrieved source material; the model is instructed never to rely on parametric knowledge alone

### 4.3 Generation Parameters

| Parameter | Value | Rationale |
|---|---|---|
| Model | gemini-2.5-flash | Cost-efficient, capable model suitable for RAG-based Q&A tasks |
| Temperature | 0.4 | Low enough to keep responses grounded; high enough to avoid robotic tone |
| Max Tokens | 900 | Sufficient for detailed career advice without unnecessary verbosity |
| Top-k Retrieval (k) | 6 | Returns enough context diversity while staying within token budget |
| Streaming | Disabled | Full response returned once completely generated; simplifies integration with Gemini API |

### 4.4 Hallucination Prevention Strategy

The system prompt instructs the model to answer exclusively from the provided context and to acknowledge gaps transparently. This is architecturally reinforced: the augmented prompt passed to the LLM explicitly labels the retrieved passages as the only trusted source, and the model is instructed to state clearly when information is unavailable rather than speculate. Source deduplication in `retriever.py` further ensures that citations presented to the user accurately reflect the chunks used.

---

## 5. Application Features and Interface

### 5.1 Example Prompt Buttons

To reduce the friction of a blank chat input and guide users toward high-value queries, five example prompt buttons are rendered at the top of the interface:

- What are the top skills for 2026?
- How do I build a strong data science portfolio?
- How do I prepare for a Data Science interview?
- What salary can I expect as a junior data scientist?
- Best roadmap to become a data scientist in 2026?

Clicking any button injects the prompt directly into the chat pipeline via Streamlit session state, bypassing the text input field for a seamless one-click experience.

### 5.2 Source Attribution

Every chatbot response displays a collapsible **Sources** section listing the specific document URLs from which the answer was drawn. This is implemented by extracting the source metadata attached to each retrieved chunk in `retriever.py` and rendering it as a formatted reference list beneath the LLM output. Source URLs are deduplicated to avoid redundant citations when multiple chunks originate from the same document. Source attribution serves two purposes: it enables users to independently verify information, and it reinforces trust by demonstrating that responses are not invented.

### 5.3 Chat Interface

The Streamlit-based interface provides a clean conversational layout with:

- Persistent message history within a session
- A text input field at the bottom
- Full response display once generation is complete
- An expandable sources panel beneath each assistant response

The interface is configured in centered-layout mode for readability on standard screens.

### 5.4 Auto-Ingestion

On startup, `app.py` checks for the existence of the `chroma_db` directory. If absent, it automatically extracts the pre-built vector database from a compressed ZIP file hosted on GitHub, making the knowledge base immediately available without triggering the ingestion pipeline. This approach ensures faster and more reliable cold starts in deployment environments.

---

## 6. Evaluation and Limitations

### 6.1 Evaluation Methodology

System performance was assessed qualitatively across three dimensions: **groundedness** (does the response derive from retrieved documents rather than parametric knowledge?), **relevance** (does the response directly address the user's question?), and **attribution accuracy** (are the cited sources the actual sources of the information in the response?). A good number of test set of representative career questions spanning all document categories was used.

### 6.2 Sample Test Questions

- What programming languages should I learn for data science in 2026?
- How do I transition into data science from a finance background?
- What is the average salary for a mid-level data scientist?
- What projects should I include in my data science portfolio?
- How do I prepare for a machine learning system design interview?

### 6.3 Results Summary

| Evaluation Dimension | Observation | Rating |
|---|---|---|
| Groundedness | Responses consistently referenced retrieved content; no hallucinated facts detected | ✅ Strong |
| Relevance | Top-k retrieval returned appropriate chunks for test questions | ✅ Strong |
| Attribution Accuracy | Source URLs matched retrieved chunks in all tested cases | ✅ Strong |
| Tone Consistency | Mentor persona maintained across all question types including edge cases | ✅ Strong |
| Out-of-scope Handling | Bot correctly declined 3 out-of-domain questions without fabricating answers | ✅ Strong |

### 6.4 Known Limitations

- The knowledge base is bounded by curated documents and cannot answer questions outside this corpus
- Web-scraped content quality is variable; JavaScript-rendered pages return little usable text via BeautifulSoup
- ChromaDB is a local store not designed for high-concurrency production deployment
- The system has no cross-session memory; each conversation begins without context from prior interactions
- The ingestion pipeline re-downloads all sources on each run, which can be slow for large document sets

### 6.5 Recommended Future Improvements

- Add a re-ranking step using a cross-encoder model to improve retrieval precision beyond cosine similarity
- Implement conversational memory using a session-scoped summary buffer for multi-turn questions
- Establish a quarterly document refresh schedule to keep industry data current
- Migrate to Pinecone or Weaviate for scalable cloud-hosted vector search
- Integrate a RAGAS evaluation framework for quantitative measurement of faithfulness and answer relevance
- Support JavaScript-rendered pages using Playwright or Selenium for richer web scraping

---

## 7. Deployment

### 7.1 Local Development Setup

To run the application locally, the following steps are required:

1. Clone the project repository and navigate to the project directory.
2. Install all dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.streamlit/secrets.toml` file containing:
   ```toml
   Gemini_API_Key = "AI..."
   ```
4. Launch the application:
   ```bash
   streamlit run app.py
   ```
   *(The ingestion pipeline runs automatically on first launch.)*

### 7.2 Streamlit Cloud Deployment

For a permanent, shareable deployment the application is hosted on Streamlit Community Cloud connected to a GitHub repository. The deployment process is:

- Push all project files (`app.py`, `ingest.py`, `retriever.py`, `sources.py`, `requirements.txt`) to a public GitHub repository
- Connect the repository to Streamlit Community Cloud at [share.streamlit.io](https://share.streamlit.io)
- Add the Gemini Key as a secret in the Streamlit Cloud dashboard under **App Settings → Secrets**
- Deploy (Streamlit Cloud installs dependencies and runs the app automatically)

This approach provides a stable, permanent public URL suitable for academic demonstration and peer review without requiring ngrok or any local infrastructure.

### 7.3 Dependencies

All required packages and their pinned versions are listed in `requirements.txt` to ensure reproducibility across environments:

| Package | Version | Purpose |
|---|---|---|
| streamlit | 1.33.0 | Web application framework and UI |
| langchain | 0.1.20 | RAG pipeline orchestration |
| langchain-community | 0.0.38 | Community integrations including ChromaDB |
| sentence-transformers | 2.7.0 | HuggingFace embedding model for chunk and query vectorisation |
| chromadb | 0.4.24 | Local vector store for chunk storage and retrieval |
| google-generativeai | 0.5.4 | Gemini API client for LLM generation |
| beautifulsoup4 | 4.12.3 | HTML parsing for web scraping |
| pypdf | 4.2.0 | PDF text extraction |
| transformers | 4.41.2 | HuggingFace transformers library |
| protobuf | 3.20.3 | Protocol buffers (ChromaDB compatibility) |
| torch | 2.2.2 | Required backend for sentence-transformers |

---

## 8. Group Contributions

| Group Member | Role | Key Contributions |
|---|---|---|
| Peter Gba | Application Developer & Technical Writer (Contributor) | Developed the full Streamlit application (`app.py`), including the chat interface, example prompt buttons, response display, source attribution, and CareerAI persona integration. Authored Section 5. |
| Amina Isa | Data Ingestion Engineer & Technical Writer (Contributor) | Designed and implemented the document ingestion pipeline (`ingest.py`), including PDF loading, web scraping, text chunking strategy, embedding, and ChromaDB vector store setup. Authored Sections 2, 3, 7. |
| Hauwa'u Al-Bashir | Retrieval Engineer & Technical Writer (Main) | Built the retrieval module (`retriever.py`), curated document sources (`sources.py`). Authored Sections 4 and 6. |
| Nusaiba Abdulnasir | — | Member was excused from active participation due to personal circumstances. |

---

## 9. Conclusion

This project demonstrates that a RAG-based architecture can deliver reliable, trustworthy, and encouraging career advice without any model fine-tuning. Through the process of combining deliberate document curation across 31 sources, semantic chunking with optimised 800-character segments, cosine similarity-based retrieval, and persona-driven prompt engineering, the system provides a grounded AI mentor that aspiring data scientists and technology professionals can genuinely rely on.

The source attribution feature is a key differentiator. It maintains transparency, builds user trust, and transforms the chatbot from a black-box oracle into a verifiable guidance tool. The architecture is modular and extensible — the knowledge base, retrieval parameters, and persona can all be updated independently as the industry evolves.

The no-fine-tuning constraint, far from being a limitation, validates a practical and scalable approach to domain-specific AI: with carefully curated retrieval and deliberate prompt engineering, a general-purpose language model can be shaped into a reliable, focused expert advisor.
