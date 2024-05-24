import torch
from sentence_transformers import SentenceTransformer, util

class BertProccessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.corpus = self.create_corpus()
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus_embeddings = self.create_corpus_embeddings()


    def create_corpus(self):
        corpus = []
        for item in self.dataset:
            if item['query_type'] == 'entity':
                text = ' '.join(item['passages']['passage_text'])
                corpus.append(text.strip())
        return corpus
    def create_corpus_embeddings(self):
        corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)
        return corpus_embeddings
    def _similarity(self, query_embeddings):
        return util.cos_sim(query_embeddings, self.corpus_embeddings)[0]
    def ranking(self, query, top_k=10):
        query_embeddings = self.model.encode(query, convert_to_tensor=True)
        cos_scores = self._similarity(query_embeddings)
        top_results = torch.topk(cos_scores, k=top_k)
        lst = []
        for score, idx in zip(top_results[0], top_results[1]):
            lst.append((idx.item(), score.item()))
        return lst