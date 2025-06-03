import json
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
client = OpenAI(api_key=api_key)
client.chat.completions.create(
     model="gpt-3.5-turbo",
     messages=[{"role": "user", "content": "Test"}],
     max_tokens=5
)

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


# --- HyDE Core Functions ---
def generate_hypothetical_answer(query: str) -> str:
    """Generate a hypothetical answer using GPT without context."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "Generate a hypothetical answer to this question as if you knew the correct information. Include key terms that would appear in relevant documents."},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Hypothetical answer generation failed: {e}")
        return ""

# Receiving embeddings
def get_embeddings(texts: List[str]) -> np.ndarray:
    """Get embeddings for texts using OpenAI."""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-ada-002"
    )
    return np.array([item.embedding for item in response.data])


def retrieve_documents(query: str, hypothetical_answer: str, documents: List[str], doc_embeddings: np.ndarray,k: int = 5) -> List[Dict]:
    """Retrieve documents using HyDE approach."""
    try:
        # Get embeddings for both query and hypothetical answer
        query_embed = get_embeddings([query])[0]
        hyde_embed = get_embeddings([hypothetical_answer])[0]

        # Blend query and HyDE embeddings (50/50 weight)
        combined_embed = 0.5 * query_embed + 0.5 * hyde_embed

        # Calculate similarity
        scores = cosine_similarity([combined_embed], doc_embeddings)[0]
        top_indices = np.argsort(scores)[::-1][:k]

        return [{
            "doc": documents[idx],
            "score": scores[idx],
            "is_hypothetical": scores[idx] > 0.5  # Threshold for HyDE influence
        } for idx in top_indices]
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []


# --- Answer Generation ---
def generate_final_answer(query: str, retrieved_docs: List[Dict]) -> Tuple[str, float]:
    """Generate final answer using retrieved documents."""
    try:
        context = "\n".join([f"Document {i + 1}: {doc['doc'][:200]}..."
                             for i, doc in enumerate(retrieved_docs)])

        prompt = f"""Answer the question using ONLY these documents. Be precise.

        Question: {query}

        Relevant Documents:
        {context}

        Answer:"""

        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200
        )
        latency = time.time() - start_time

        return response.choices[0].message.content.strip(), latency
    except Exception as e:
        print(f"Answer generation failed: {e}")
        return "Error generating answer", 0.0


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
def run_hyde_experiment(questions: List[Dict], documents: List[str]) -> List[Dict]:
    """Run end-to-end HyDE RAG experiment."""
    # Pre-compute document embeddings
    doc_embeddings = get_embeddings(documents)
    results = []

    for q in questions:
        query = q["question"]
        reference = q["answer"]
        relevant_docs = q["relevant_docs"]

        # HyDE Step 1: Generate hypothetical answer
        hyde_answer = generate_hypothetical_answer(query)

        # HyDE Step 2: Retrieve documents
        retrieved_docs = retrieve_documents(query, hyde_answer, documents, doc_embeddings)

        # Generate final answer
        final_answer, latency = generate_final_answer(query, retrieved_docs)

        # Evaluate
        metrics = evaluate_metrics(final_answer, reference, retrieved_docs, relevant_docs)
        metrics["latency"] = latency

        results.append({
            "question": query,
            "hypothetical_answer": hyde_answer,
            "generated_answer": final_answer,
            "reference_answer": reference,
            "retrieved_docs": [doc["doc"] for doc in retrieved_docs],
            "metrics": metrics
        })

    return results


# Main function
def main():
    results = run_hyde_experiment(SAMPLE_QUESTIONS, DOCUMENTS)

    with open("hyde_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for res in results:
        print(f"Question: {res['question']}")
        print(f"Answer: {res['generated_answer']}")
        print(
            f"Metrics: ROUGE-L={res['metrics']['rouge_l']:.3f}, BLEU={res['metrics']['bleu']:.3f}, F1 Score={res['metrics']['f1_score']:.3f}, Latency={res['metrics']['latency']:.3f}s")
        print("-" * 80)



if __name__ == "__main__":
    main()

