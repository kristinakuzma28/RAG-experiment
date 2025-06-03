import json
import os
import time
from typing import List, Dict, Tuple
from openai import OpenAI
from PyPDF2 import PdfReader
from rouge_score import rouge_scorer
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics.pairwise import cosine_similarity

# Configure OpenAI API
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY was not found.")
    exit(1)

try:
    client = OpenAI(api_key=api_key)
    client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": "Test"}],
        max_tokens=5
    )
    print("API key successfully verified.")
except Exception as e:
    print(f"Error verifying OpenAI API: {e}. Check your key and model access.")
    exit(1)

# Initialize stemmer
stemmer = PorterStemmer()

# Document corpus
def read_pdfs_from_folder(folder_path: str) -> List[str]:
    documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                reader = PdfReader(file_path)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                documents.append(text.strip())
            except Exception as e:
                print(f"Помилка при читанні {filename}: {e}")
    return documents

PDF_FOLDER = "./hotpotqa_pdfs"
DOCUMENTS = read_pdfs_from_folder(PDF_FOLDER)

# Sample questions
SAMPLE_QUESTIONS = [
    {
        "question": "Who directed the film about Ed Wood's life, and what was the name of the actor who played Bela Lugosi in it?",
        "answer": "The film about Ed Wood's life was directed by Tim Burton, and Martin Landau played Bela Lugosi.",
        "relevant_docs": ["Ed Wood (film).pdf"]
    },
    {
        "question": "Which actress starred in both 'Kiss and Tell' and its sequel, and what was the role she played?",
        "answer": "Shirley Temple starred in both 'Kiss and Tell' and its sequel 'A Kiss for Corliss' as Corliss Archer.",
        "relevant_docs": [
            "Kiss and Tell (1945 film).pdf",
            "A Kiss for Corliss.pdf"
        ]
    },
    {
        "question": "Who worked as an editor on 'Meet Corliss Archer' and what was his nationality?",
        "answer": "Charles Craft, an English-born American, worked as an editor on 'Meet Corliss Archer'.",
        "relevant_docs": ["Charles Craft.pdf "]
    },
    {
        "question": "Who directed the film 'Doctor Strange' in 2016, and which actors played the main roles?",
        "answer": "The film 'Doctor Strange' in 2016 was directed by Scott Derrickson, and the main roles were played by Benedict Cumberbatch, Chiwetel Ejiofor, and Rachel McAdams.",
        "relevant_docs": ["Doctor Strange (2016 film).pdf"]
    },
    {
        "question": "Which actor appeared in Ed Wood films and what role did he play in 'Plan 9 from Outer Space'?",
        "answer": "Conrad Brooks appeared in Ed Wood's films and played a role as an actor in 'Plan 9 from Outer Space'.",
        "relevant_docs": ["Conrad Brooks.pdf"]
    },
    {
        "question": "Who directed 'Hellraiser: Inferno', and was it the first 'Hellraiser' film released straight-to-DVD?",
        "answer": "The director of 'Hellraiser: Inferno' was Scott Derrickson, and it was the first 'Hellraiser' film released straight-to-DVD.",
        "relevant_docs": ["Hellraiser_Inferno.pdf"]
    },
    {
        "question": "Which actress voiced Judy Jetson and played the title role in the radio program 'Meet Corliss Archer'?",
        "answer": "Janet Waldo voiced Judy Jetson and played the title role in the radio program 'Meet Corliss Archer'.",
        "relevant_docs": [
            "Janet Waldo.pdf",
            "Meet Corliss Archer.pdf"
        ]
    },
    {
        "question": "Who directed 'The Exorcism of Emily Rose', and which actress played the main role of the defense counsel?",
        "answer": "'The Exorcism of Emily Rose' was directed by Scott Derrickson, and the main role of the defense counsel was played by Laura Linney.",
        "relevant_docs": ["The Exorcism of Emily Rose.pdf"]
    },
    {
        "question": "Which composer worked on the music for 'John Wick' and collaborated with director Scott Derrickson?",
        "answer": "Tyler Bates worked on the music for 'John Wick' and collaborated with director Scott Derrickson.",
        "relevant_docs": ["Tyler Bates.pdf"]
    },
    {
        "question": "Which film starring Shirley Temple is a sequel to 'Kiss and Tell', and who was its director?",
        "answer": "The film 'A Kiss for Corliss' is a sequel to 'Kiss and Tell', and its director was Richard Wallace.",
        "relevant_docs": [
            "A Kiss for Corliss.pdf",
            "Kiss and Tell (1945 film).pdf"
        ]
    }
]

# --- Query Decomposition Functions ---
def decompose_query(query: str) -> List[str]:
    """Break down complex questions into sub-questions using GPT."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Split complex questions into standalone sub-questions. Example: 'Who directed X and who starred?' → ['Who directed X?', 'Who starred in X?']"},
                {"role": "user", "content": f"Decompose this question: {query}"}
            ],
            temperature=0.3,
            max_tokens=150
        )
        decomposed = response.choices[0].message.content.strip().split("\n")
        return [q.strip().rstrip('?') + '?' for q in decomposed if q.strip()]
    except Exception as e:
        print(f"Decomposition error: {e}")
        return [query]  # Fallback to original query

def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for texts using OpenAI."""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return np.array([item.embedding for item in response.data])


def retrieve_documents(query: str, documents: List[str], doc_embeddings: np.ndarray, k: int = 3) -> List[Dict]:
    """Retrieve relevant documents for a sub-question."""
    query_embedding = get_embeddings([query.lower()])[0]
    scores = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    return [{"doc": documents[idx], "score": scores[idx]} for idx in top_indices]


# --- Answer Generation ---
def generate_subanswers(subqueries: List[str], documents: List[str], doc_embeddings: np.ndarray) -> List[Dict]:
    """Generate answers for each sub-question."""
    subanswers = []
    for subquery in subqueries:
        retrieved_docs = retrieve_documents(subquery, documents, doc_embeddings)
        answer, _ = generate_answer(subquery, retrieved_docs)
        subanswers.append({
            "subquery": subquery,
            "answer": answer,
            "docs": retrieved_docs
        })
    return subanswers


def generate_answer(query: str, documents: List[Dict]) -> Tuple[str, float]:
    """Generate answer using retrieved documents."""
    try:
        context = "\n".join([f"Document {i + 1}: {doc['doc'][:200]}..." for i, doc in enumerate(documents)])
        prompt = f"""Answer the question using ONLY the provided context. Be concise.

        Context:
        {context}

        Question: {query}
        Answer:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150
        )
        return response.choices[0].message.content.strip(), 0.0
    except Exception as e:
        print(f"Generation error: {e}")
        return "Error generating answer", 0.0


def combine_answers(subanswers: List[Dict], original_query: str) -> str:
    """Combine sub-answers into a final coherent answer."""
    context = "\n".join([f"Sub-question: {sa['subquery']}\nAnswer: {sa['answer']}" for sa in subanswers])
    prompt = f"""
    Combine these sub-answers into a single coherent answer for the original question.
    Original question: {original_query}

    Sub-answers:
    {context}

    Final answer:
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=200
    )
    return response.choices[0].message.content.strip()


# --- Evaluation Metrics ---
def evaluate_metrics(generated: str, reference: str, retrieved_docs: List[Dict], relevant_docs: List[str]) -> Dict:
    """Calculate evaluation metrics."""

    # Normalize texts
    def normalize(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        return re.sub(r'\s+', ' ', text).strip()

    gen_norm = normalize(generated)
    ref_norm = normalize(reference)

    # ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ref_norm, gen_norm)['rougeL'].fmeasure

    # BLEU
    bleu = sentence_bleu([ref_norm.split()], gen_norm.split(),
                         smoothing_function=SmoothingFunction().method1)

    # Document Retrieval F1
    retrieved_texts = [normalize(doc['doc']) for doc in retrieved_docs]
    relevant_texts = [normalize(doc) for doc in relevant_docs]

    tp = sum(1 for ret in retrieved_texts if any(rel in ret or ret in rel for rel in relevant_texts))
    precision = tp / len(retrieved_texts) if retrieved_texts else 0
    recall = tp / len(relevant_texts) if relevant_texts else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "rouge_l": rouge_l,
        "bleu": bleu,
        "f1_score": f1
    }


# --- Main Experiment ---
def run_experiment(questions: List[Dict], documents: List[str]) -> List[Dict]:
    """Run end-to-end query decomposition RAG experiment."""
    doc_embeddings = get_embeddings(documents)
    results = []

    for q in questions:
        start_time = time.time()
        query = q["question"]
        reference = q["answer"]
        relevant_docs = q["relevant_docs"]

        # Step 1: Query Decomposition
        subqueries = decompose_query(query)
        print(f"Decomposed: {query} → {subqueries}")

        # Step 2: Sub-question Processing
        subanswers = generate_subanswers(subqueries, documents, doc_embeddings)

        # Step 3: Answer Combination
        final_answer = combine_answers(subanswers, query)
        latency = time.time() - start_time

        # Evaluation
        all_retrieved_docs = [doc for sa in subanswers for doc in sa["docs"]]
        metrics = evaluate_metrics(final_answer, reference, all_retrieved_docs, relevant_docs)
        metrics["latency"] = latency

        results.append({
            "question": query,
            "subqueries": subqueries,
            "generated_answer": final_answer,
            "reference_answer": reference,
            "metrics": metrics,
            "retrieved_docs": [doc["doc"] for doc in all_retrieved_docs]
        })

    return results


# Main function
def main():
    results = run_experiment(SAMPLE_QUESTIONS, DOCUMENTS)

    with open("query_decomposition_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for res in results:
        print(f"Question: {res['question']}")
        print(f"Answer: {res['generated_answer']}")
        print(
            f"Metrics: ROUGE-L={res['metrics']['rouge_l']:.3f}, BLEU={res['metrics']['bleu']:.3f}, F1 Score={res['metrics']['f1_score']:.3f}, Latency={res['metrics']['latency']:.3f}s")
        print("-" * 80)



if __name__ == "__main__":
    main()

