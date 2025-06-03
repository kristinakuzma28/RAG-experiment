# RAG-experiment

## Overview

This project implements various **Retrieval-Augmented Generation (RAG)** methods to answer questions based on a corpus of documents, utilizing the *HotpotQA dataset*. It includes different retrieval and generation techniques such as BM25, Semantic Search, Query Decomposition, Hybrid RAG, and HyDE. The project evaluates performance using metrics like **ROUGE-L**, **BLEU**, **F1-score**, and *latency*, and generates visualizations to compare these methods. Additionally, it supports PDF document creation from the dataset.

## Project Structure

- **Base_RAG**:
  - `BM25`:
    - `bm25_results.json`: Results of the BM25 RAG method.
    - `bm25.py`: Script for BM25-based RAG pipeline.
  - `Semantic_Search`:
    - `semantic_search_results.json`: Results of semantic search RAG method.
    - `semantic_search.py`: Script for semantic search implementation.

- **data**:
  - `hotpotqa_pdfs`: Directory for generated PDF files.
  - `hotpot_dev_fullwiki_v1.json`: HotpotQA dataset file.
  - `main.py`: Script for generating PDF documents.
  - `pdf_creation_log.txt`: Log file for PDF creation process.

- **Hybrid_RAG**:
  - `hybrid_rag_results.json`: Results of the Hybrid RAG method.
  - `hybrid_rag.py`: Script for Hybrid RAG implementation.

- **HyDE**:
  - `hyde_results.json`: Results of the HyDE method.
  - `hyde.py`: Script for HyDE implementation.

- **Query_Decomposition**:
  - `query_decomposition_results.json`: Results of the Query Decomposition method.
  - `query_decomposition.py`: Script for Query Decomposition implementation.
  - `plot.ipynb`: Jupyter notebook for visualizing metrics.

## Requirements

To run the project, install the following dependencies:
```bash
pip install openai rank-bm25 rouge-score numpy nltk reportlab pandas matplotlib jupyter
```
Additionally, set up an OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY='your-api-key'
```


## Usage

1. **Run BM25 RAG (`bm25.py`)**:
   - Executes question answering using BM25 retrieval and OpenAI API.
   - Saves results to `base_rag_results.json`.
```bash
python bm25.py
```
2. **Run Semantic Search (`semantic_search.py`)**:
   - Implements semantic search-based RAG and saves results to `semantic_search_results.json`.
```bash
python semantic_search.py
```
3. **Run Query Decomposition (`query_decomposition.py`)**:
   - Decomposes queries into subqueries for improved RAG and saves results to  `query_decomposition_results.json`.
``` bask
python query_decomposition.py
```
4. **Run Hybrid RAG (`hybrid_rag.py`)**:
   - Combines multiple RAG techniques and saves results to `hybrid_rag_results.json`.
```bash
python hybrid_rag.py
```
5. **Run HyDE (`hyde.py`)**:
   - Implements Hypothetical Document Embeddings (HyDE) and saves results to `hyde_results.json`.
```bash
python hyde.py
```
6. **Generate PDF Documents (`main.py`)**:
   - Processes the HotpotQA dataset (`hotpot_dev_fullwiki_v1.json`) to create PDF files for up to 20 unique documents.
   - Outputs PDFs to the `./hotpotqa_pdfs directory` and saves a mapping to `document_mapping.csv`.
```bash
python main.py
```
7. **Visualize Metrics (`plot.ipynb`)**:
   - Generates bar charts comparing ROUGE-L, BLEU, F1-score, and latency for different RAG methods.
```bash
jupyter notebook plot.ipynb
```


## Output

   - **JSON Results**: The `bm25_results.json`, `semantic_search_results.json`, `query_decomposition_results.json`, `hybrid_rag_results.json`, and `hyde_results.json` files contain detailed results for each question, including generated answers, reference answers, retrieved documents, and metrics (ROUGE-L, BLEU, F1-score, latency).

   - **PDF Documents**: Up to 20 PDF files are generated in the `./hotpotqa_pdfs` directory, with a log of the process in `pdf_creation_log.txt`.

   - **Visualizations**: Bar charts comparing the performance of different RAG methods are displayed when running `plot.ipynb`.

## Notes
   - The project assumes the presence of a `hotpot_dev_fullwiki_v1.json` file for PDF generation. Ensure this file is available in the `data` directory or update the path in `main.py`.

   - The evaluation metrics in `bm25.py` use normalized text for consistency, and BM25 is used for document retrieval.

   - The visualization script (`plot.ipynb`) compares five RAG methods, but ensure all result files are updated with actual metrics.

