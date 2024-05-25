from datasets import load_dataset
from nltk.corpus import stopwords
import string
import numpy as np
from nltk.stem import PorterStemmer

class BoWProcessor:
    def __init__(self, dataset):
        self.dataset = dataset
        self.corpus = self.create_corpus()
        self.vocab = self.create_vocabulary()
        self.doc_matrix = self.create_BoW_doc_matrix()

    def create_corpus(self):
        corpus = []
        for item in self.dataset:
            if item['query_type'] == 'entity':
                text = ''
                for content in item['passages']['passage_text']:
                    text += content + ''
                corpus.append(text.strip())
        return corpus

    def tokenize(self, doc):
        return doc.split()

    def lowercase_text(self, text):
        return text.lower()

    def remove_punctuation(self, text):
        remove_word = string.punctuation
        for ch in remove_word:
            text = text.replace(ch, '')
        return text

    def remove_stopWord(self, text):
        stopwords_lst = stopwords.words('english')
        for ch in stopwords_lst:
            text = text.replace(ch, '')
        return text.strip()
    def stemming(self,text):
        stemmer = PorterStemmer()
        tokens = self.tokenize(text)
        stemmed_lst = [stemmer.stem(token) for token in tokens]
        return ' '.join(stemmed_lst)

    def text_normalize(self, text):
        text = self.lowercase_text(text)
        text = self.remove_punctuation(text)
        text = self.remove_stopWord(text)
        text = self.stemming(text)
        return text

    def create_vocabulary(self):
        vocab = set()
        for doc in self.corpus:
            normalized_doc = self.text_normalize(doc)
            tokens = self.tokenize(normalized_doc)
            vocab.update(tokens)
        return list(vocab)

    def BoW_vectorize(self, text):
        word_dict = {word: 0 for word in self.vocab}
        text = self.text_normalize(text)
        tokens = self.tokenize(text)
        for token in tokens:
            if token in word_dict:
                word_dict[token] += 1
        return list(word_dict.values())

    def create_BoW_doc_matrix(self):
        doc_matrix = {}
        for idx, doc in enumerate(self.corpus):
            vector = self.BoW_vectorize(doc)
            doc_matrix[idx] = vector
        return doc_matrix

    def cosine_similarity(self, a, b):
        a = np.array(a)
        b = np.array(b)
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity = dot_product / (norm_a * norm_b)
        return similarity

    def ranking(self, query, topk):
        query_vector = self.BoW_vectorize(query)
        scores = []
        for idx, doc_vector in self.doc_matrix.items():
            similarity = self.cosine_similarity(query_vector, doc_vector)
            scores.append((idx, similarity))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]


# Load dataset
# dataset = load_dataset('ms_marco', 'v1.1')['train']
#
# # Initialize BoWProcessor
# processor = BoWProcessor(dataset)
#
# # Define query and topk
# query = 'what is the official language in Fiji'
# topk = 10
#
# # Get rankings
# rankings = processor.ranking(query, topk)
#
# # Display results
# print(f'Query: {query}')
# print('=== Relevant docs ===')
# for idx, (doc_idx, score) in enumerate(rankings):
#     print(f'Top {idx + 1}; Score: {score:.4f}')
#     print(processor.corpus[doc_idx])
#     print('\n')
