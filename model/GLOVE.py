import numpy as np
import json
import string

class GloVeProcessor:
    def __init__(self, dataset, json_file):
        self.dataset = dataset
        self.glove_embeddings = self.load_glove_embeddings(json_file)
        self.corpus = self.create_corpus()
        self.doc_embeddings = self.create_doc_embeddings()

    def load_glove_embeddings(self, json_file):
        with open(json_file, 'r', encoding='utf-8') as file:
            embeddings = json.load(file)
        return {word: np.array(vector, dtype='float32') for word, vector in embeddings.items()}

    def create_corpus(self):
        corpus = []
        for item in self.dataset:
            if item['query_type'] == 'entity':
                text = ' '.join(item['passages']['passage_text'])
                corpus.append(text.strip())
        return corpus

    def tokenize(self, text):
        return text.split()

    def text_normalize(self, text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text

    def get_embedding(self, word):
        return self.glove_embeddings.get(word, np.zeros(300))

    def create_doc_embeddings(self):
        doc_embeddings = []
        for doc in self.corpus:
            tokens = self.tokenize(self.text_normalize(doc))
            if tokens:
                embeddings = [self.get_embedding(token) for token in tokens]
                doc_embedding = np.mean(embeddings, axis=0)
            else:
                doc_embedding = np.zeros(300)
            doc_embeddings.append(doc_embedding)
        return doc_embeddings

    def get_query_embedding(self, query):
        tokens = self.tokenize(self.text_normalize(query))
        embeddings = [self.get_embedding(token) for token in tokens]
        if embeddings:
            query_embedding = np.mean(embeddings, axis=0)
        else:
            query_embedding = np.zeros(300)
        return query_embedding

    def cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def ranking(self, query, topk):
        query_embedding = self.get_query_embedding(query)
        scores = []
        for idx, doc_embedding in enumerate(self.doc_embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            scores.append((idx, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]
