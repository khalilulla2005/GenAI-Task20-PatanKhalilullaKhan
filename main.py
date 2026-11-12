import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from src.vectorization import manual_one_hot

def run_assignment():
    df = pd.read_csv('data/dataset.csv')
    text_data = df['text']
    sentences = text_data.tolist()[:5]

    # --- Task 1 & 2 (Previous) ---
    print("--- Task 1 & 2: One-Hot Encoding ---")
    vocab, matrix = manual_one_hot(sentences)
    print("Manual OHE Vocabulary:", vocab)
    
    vec = CountVectorizer(binary=True)
    ohe = vec.fit_transform(sentences)
    print("Scikit-learn OHE Matrix shape:", ohe.toarray().shape)

    # --- Task 3 & 4: BoW & Frequency Analysis ---
    print("\n--- Task 3 & 4: BoW & Frequency ---")
    bow_vec = CountVectorizer()
    bow_matrix = bow_vec.fit_transform(text_data)
    
    # Frequency analysis
    word_freq = np.asarray(bow_matrix.sum(axis=0)).ravel()
    vocab = bow_vec.get_feature_names_out()
    freq_df = pd.DataFrame({'word': vocab, 'freq': word_freq})
    
    print("Top 10 frequent:", freq_df.sort_values(by='freq', ascending=False).head(10)['word'].tolist())
    print("Top 10 least frequent:", freq_df.sort_values(by='freq', ascending=True).head(10)['word'].tolist())

    # --- Task 5 & 6: N-grams ---
    print("\n--- Task 5 & 6: N-grams ---")
    for n in [(1,1), (2,2), (3,3), (1,2)]:
        vec = CountVectorizer(ngram_range=n)
        vec.fit_transform(text_data)
        print(f"N-gram {n} vocab size: {len(vec.get_feature_names_out())}")

    # --- Task 7 & 8: TF-IDF & Comparison ---
    print("\n--- Task 7 & 8: TF-IDF Comparison ---")
    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(text_data)
    
    # Simple term comparison
    term = 'product'
    if term in bow_vec.get_feature_names_out():
        idx = bow_vec.vocabulary_.get(term)
        print(f"Term '{term}': BoW={bow_matrix[:, idx].sum()}, TF-IDF={tfidf_matrix[:, idx].sum():.4f}")

    # --- Task 9: Hyperparameter exploration ---
    print("\n--- Task 9: Hyperparameter Exploration ---")
    vec_params = CountVectorizer(max_features=5, min_df=2)
    vec_params.fit_transform(text_data)
    print("Vocab size (max_features=5, min_df=2):", len(vec_params.get_feature_names_out()))

if __name__ == "__main__":
    run_assignment()
