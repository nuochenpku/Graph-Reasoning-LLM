import distance
from typing import Optional, Dict, Sequence, List

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from scipy.linalg import norm

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
    
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 求交集
    numerator = np.sum(np.min(vectors, axis=0))
    # 求并集
    denominator = np.sum(np.max(vectors, axis=0))
    # 计算杰卡德系数
    return 1.0 * numerator / denominator


def sorted_jaccard(strings: List, target: str):
    results = sorted(strings, key=lambda x: jaccard_similarity(filter_string(x), target))
    results_indices = [strings.index(result) for result in results]
    return results_indices[::-1]

def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    # 将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    # 转化为TF矩阵
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1, s2]
    vectors = cv.fit_transform(corpus).toarray()
    # 计算TF系数
    return np.dot(vectors[0], vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))


def sorted_tfidf(strings: List, target: str):
    results = sorted(strings, key=lambda x: tfidf_similarity(filter_string(x), target))
    results_indices = [strings.index(result) for result in results]
    return results_indices[::-1]


