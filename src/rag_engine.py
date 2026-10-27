import os
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGEngine:
    """
    Retrieval-Augmented Generation (RAG) vector retrieval engine.
    Indexes product FAQs across 10 categories, performs vector similarity search,
    and returns relevant context chunks for LLM prompt augmentation.
    """
    def __init__(self, faq_file_path: str = None):
        if faq_file_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            faq_file_path = os.path.join(base_dir, 'knowledge_base', 'faqs.json')
            
        self.faq_file_path = faq_file_path
        self.documents = []
        self.doc_texts = []
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english')
        self._load_and_index()

    def _load_and_index(self):
        with open(self.faq_file_path, 'r', encoding='utf-8') as f:
            self.documents = json.load(f)

        # Index question and answer text together for dense keyword/semantic coverage
        self.doc_texts = [
            f"{doc['category']} {doc['question']} {doc['answer']}" for doc in self.documents
        ]
        self.tfidf_matrix = self.vectorizer.fit_transform(self.doc_texts)

    def retrieve(self, query: str, category_filter: str = None, top_k: int = 2):
        """
        Retrieves top_k relevant FAQ documents.
        Applies optional category boosting / filtering.
        """
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        results = []
        for idx, sim in enumerate(similarities):
            doc = self.documents[idx]
            score = float(sim)
            # Boost score if category matches classified intent
            if category_filter and doc['category'] == category_filter:
                score *= 1.25
            results.append({
                'id': doc['id'],
                'category': doc['category'],
                'question': doc['question'],
                'answer': doc['answer'],
                'similarity_score': round(min(1.0, score), 4)
            })

        # Sort descending by score
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return results[:top_k]

    def format_context(self, retrieved_docs):
        """
        Formats retrieved FAQ documents into a clean string for LLM prompt insertion.
        """
        if not retrieved_docs:
            return "No specific knowledge base entry matched."
            
        context_parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            context_parts.append(
                f"[Doc {i} | Category: {doc['category'].upper()} | Relevance: {doc['similarity_score']:.2f}]\n"
                f"Question: {doc['question']}\n"
                f"Answer: {doc['answer']}"
            )
        return "\n\n".join(context_parts)
