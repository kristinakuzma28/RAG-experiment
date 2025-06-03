import json
import time
import os
from typing import List, Dict, Tuple
from openai import OpenAI
from PyPDF2 import PdfReader
from rouge_score import rouge_scorer
import numpy as np
import re
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
# Receiving embeddings
def get_embeddings(texts: List[str]) -> np.ndarray:
    try:
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-ada-002"
        )
        return np.array([item.embedding for item in response.data])
    except Exception as e:
        print(f"Error getting embeddings: {e}")
        return np.zeros((len(texts), 1536))

DOCUMENT_EMBEDDINGS = get_embeddings(DOCUMENTS)

# Semantic search (based on embeddings)
def retrieve_documents_semantic(query: str, documents: List[str], doc_embeddings: np.ndarray, k: int = 5) -> List[Dict]:
    try:
        query_embedding = get_embeddings([query.lower()])[0]
        semantic_scores = cosine_similarity([query_embedding], doc_embeddings)[0]
        top_k_indices = np.argsort(semantic_scores)[::-1][:k]
        return [{"doc": documents[idx], "score": semantic_scores[idx]} for idx in top_k_indices]
    except Exception as e:
        print(f"Error in semantic retrieval: {e}")
        return []


# Generating a response using the OpenAI API
def generate_answer_openai(query: str, documents: List[Dict]) -> Tuple[str, float]:
    start_time = time.time()
    context = "\n".join([f"Document {i + 1}: {doc['doc'][:100]}..." for i, doc in enumerate(documents)])
    prompt = (
        f"You are an expert in answering questions with high accuracy. "
        f"Carefully read the provided documents and extract the exact information to answer the question. "
        f"Do not export or add details beyond the context. Use concise phrasing that matches the expected answer format.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Answer strictly based on the context, using precise and matching phrasing."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.3
    )
    latency = time.time() - start_time
    return response.choices[0].message.content.strip(), latency

# Evaluation of results
def evaluate_metrics(generated: str, reference: str, retrieved_docs: List[Dict], relevant_docs: List[str], query: str) -> Dict:
    def normalize_text(text: str) -> str:
        text = re.sub(r'[^\w\s]', '', text.lower())
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    # Normalize texts
    gen_norm = normalize_text(generated)
    ref_norm = normalize_text(reference)

    # 1. Calculate ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ref_norm, gen_norm)['rougeL'].fmeasure

    # 2. Calculate BLEU
    gen_tokens = gen_norm.split()
    ref_tokens = ref_norm.split()
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=smoothie)

    # 3. Calculate F1-score for documents
    retrieved_texts = [normalize_text(doc['doc']) for doc in retrieved_docs[:5]]
    relevant_texts = [normalize_text(doc) for doc in relevant_docs]

    # Find matches between retrieved and relevant documents
    true_positives = 0
    for ret_doc in retrieved_texts:
        for rel_doc in relevant_texts:
            if rel_doc in ret_doc or ret_doc in rel_doc:
                true_positives += 1
                break

    precision = true_positives / len(retrieved_texts) if retrieved_texts else 0
    recall = true_positives / len(relevant_texts) if relevant_texts else 0

    # Calculate F1-score for documents
    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0


    return {
        "rouge_l": rouge_l,
        "bleu": bleu,
        "f1_score": f1_score
    }

# Start of the experiment
def run_base_rag_experiment(questions: List[Dict], documents: List[str], doc_embeddings: np.ndarray) -> List[Dict]:
    results = []
    for q in questions:
        query = q["question"]
        reference = q["answer"]
        relevant_docs = q["relevant_docs"]

        retrieved_docs = retrieve_documents_semantic(query, documents, doc_embeddings, k=5)

        generated_answer, latency = generate_answer_openai(query, retrieved_docs)
        metrics = evaluate_metrics(generated_answer, reference, retrieved_docs, relevant_docs, query)
        metrics["latency"] = latency

        results.append({
            "question": query,
            "generated_answer": generated_answer,
            "reference_answer": reference,
            "retrieved_docs": [doc["doc"] for doc in retrieved_docs],
            "metrics": metrics
        })
    return results


# Main function
def main():
    doc_embeddings = get_embeddings(DOCUMENTS)  # Precompute embeddings
    results = run_base_rag_experiment(SAMPLE_QUESTIONS, DOCUMENTS, doc_embeddings)

    with open("semantic_search_results.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    for res in results:
        print(f"Question: {res['question']}")
        print(f"Answer: {res['generated_answer']}")
        print(
            f"Metrics: ROUGE-L={res['metrics']['rouge_l']:.3f}, F1={res['metrics']['f1_score']:.3f}, Latency={res['metrics']['latency']:.2f}s")
        print("-" * 80)


if __name__ == "__main__":
    main()