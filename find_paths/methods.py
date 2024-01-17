import distance
from typing import Optional, Dict, Sequence, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.linalg import norm
import torch.nn as nn
from transformers import BertTokenizer, BertModel, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel
import torch

def filter_string(string: str):
    
    return string.replace('\n',' ').replace('###','')


def edit_distance(s1: str, s2: str):
    return distance.levenshtein(s1, s2)


## sorted all distances and return the indices in order from smallest to largest
def sorted_distance(strings: List, target: str):
    results = sorted(strings, key=lambda x: edit_distance(filter_string(x), target))
    results_indices = [strings.index(result) for result in results]
    return results_indices


def jaccard_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis=0))
    denominator = np.sum(np.max(vectors, axis=0))
    return 1.0 * numerator / denominator


def sorted_jaccard(strings: List, target: str):
    results = sorted(strings, key=lambda x: jaccard_similarity(filter_string(x), target))
    results_indices = [strings.index(result) for result in results]
    return results_indices[::-1]


def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    s1, s2 = add_space(s1), add_space(s2)
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def sorted_tfidf(strings: List, target: str):
    results = sorted(strings, key=lambda x: tfidf_similarity(filter_string(x), target))
    results_indices = [strings.index(result) for result in results]
    return results_indices[::-1]


class TextModel(nn.Module):
    def __init__(self, encoder):
        super(TextModel, self).__init__()
        self.encoder = encoder
        if self.encoder == 'Bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textmodel = BertModel.from_pretrained('bert-base-uncased')
        if self.encoder == 'Roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.textmodel = RobertaModel.from_pretrained('roberta-base')
        if self.encoder == 'SentenceBert':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
            self.textmodel = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
        if self.encoder == 'SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
            self.textmodel = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        if self.encoder == 'e5':
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
            self.textmodel = AutoModel.from_pretrained('intfloat/e5-base-v2')

    def forward(self, input):
        inputs = self.tokenizer(input, return_tensors='pt', truncation=True, padding=True).to(self.textmodel.device)
        with torch.no_grad():
            outputs = self.textmodel(**inputs)

        text_embedding = outputs[0][:,0,:].squeeze()
        return text_embedding
