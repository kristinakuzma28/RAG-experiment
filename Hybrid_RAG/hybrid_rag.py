import json
import os
import time
from typing import List, Dict
from openai import OpenAI
from PyPDF2 import PdfReader
from rank_bm25 import BM25Okapi
from rouge_score import rouge_scorer
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.translate.bleu_score import sentence_bleu
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
    return np.array([client.embeddings.create(input=[t], model="text-embedding-ada-002").data[0].embedding for t in texts])

DOCUMENT_EMBEDDINGS = get_embeddings(DOCUMENTS)

# Hybrid Search
def retrieve_docs(query: str, docs: List[str], embeddings: np.ndarray, k: int = 5, bm25_w: float = 0.5, sem_w: float = 0.5) -> List[str]:
    tokenized_docs = [re.findall(r'\w+', d.lower()) for d in docs]
    bm25 = BM25Okapi(tokenized_docs)
    bm25_scores = bm25.get_scores(re.findall(r'\w+', query.lower()))

    query_emb = get_embeddings([query.lower()])[0]
    sem_scores = cosine_similarity([query_emb], embeddings)[0]  # Повний масив балів

    bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-10)
    sem_norm = sem_scores / (np.max(sem_scores) + 1e-10)
    hybrid_scores = bm25_w * bm25_norm + sem_w * sem_norm

    return [docs[i] for i in np.argsort(hybrid_scores)[::-1][:k]]


# Generating a response using the OpenAI API
def generate_answer(query: str, docs: List[str]) -> tuple[str, float]:
    start_time = time.time()
    context = "\n".join([f"Doc {i + 1}: {d[:50]}..." for i, d in enumerate(docs)])
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"Context: {context}\nQ: {query}\nA:"}],
        max_tokens=50
    )
    return response.choices[0].message.content.strip(), time.time() - start_time


# Evaluation of results
def evaluate(generated: str, reference: str, retrieved: List[str], relevant: List[str]) -> Dict:
    gen, ref = re.sub(r'[^\w\s]', '', generated.lower()), re.sub(r'[^\w\s]', '', reference.lower())
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(ref, gen)['rougeL'].fmeasure
    bleu = sentence_bleu([ref.split()], gen.split())
    precision = sum(1 for r in retrieved[:5] if any(rel in r for rel in relevant)) / 5
    recall = precision / len(relevant) if relevant else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {"rouge_l": rouge_l, "bleu": bleu, "f1_score": f1}


# Start of the experiment
def run_experiment(questions: List[Dict], docs: List[str], embeddings: np.ndarray) -> List[Dict]:
    results = []
    for q in questions:
        retrieved = retrieve_docs(q["question"], docs, embeddings)
        answer, latency = generate_answer(q["question"], retrieved)
        metrics = evaluate(answer, q["answer"], retrieved, q["relevant_docs"])
        metrics["latency"] = latency
        results.append({
            "question": q["question"],
            "generated_answer": answer,
            "reference_answer": q["answer"],
            "retrieved_docs": retrieved,
            "metrics": metrics
        })
    return results

# Main function
def main():
    try:
        results = run_experiment(SAMPLE_QUESTIONS, DOCUMENTS, DOCUMENT_EMBEDDINGS)

        # Saving results in JSON
        with open("hybrid_rag_results.json", "w", encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        # Displaying the results
        for res in results:
            print(f"Question: {res['question']}")
            print(f"Generated Answer: {res['generated_answer']}")
            print(f"Reference Answer: {res['reference_answer']}")
            print(f"Metrics: ROUGE-L={res['metrics']['rouge_l']:.3f}, BLEU={res['metrics']['bleu']:.3f}, F1 Score={res['metrics']['f1_score']:.3f}, Latency={res['metrics']['latency']:.3f}s")
            print("-" * 80)
    except Exception as e:
        print(f"Error running experiment: {e}")

if __name__ == "__main__":
    main()

