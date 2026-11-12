import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Task 1: Manual One-Hot Encoding
def manual_one_hot(sentences):
    vocab = sorted(list(set(" ".join(sentences).split())))
    encoded = []
    for s in sentences:
        vec = [1 if word in s.split() else 0 for word in vocab]
        encoded.append(vec)
    return vocab, np.array(encoded)

# Task 3-9 Wrapper
class VectorizerPipeline:
    def __init__(self, data):
        self.data = data
        self.count_vec = CountVectorizer()
        self.tfidf_vec = TfidfVectorizer()

    def run_all(self):
        # BoW
        bow = self.count_vec.fit_transform(self.data)
        # TF-IDF
        tfidf = self.tfidf_vec.fit_transform(self.data)
        return bow, tfidf
