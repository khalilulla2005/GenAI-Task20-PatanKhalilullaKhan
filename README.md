# Assignment 18: Text Vectorization Techniques

**Student Name:** Patan Khalilulla Khan

## Project Structure
```text
GenAI-Task18-PatanKhalilullaKhan/
├── data/
│   └── dataset.csv          # Raw data
├── src/
│   └── vectorization.py     # Implementation of manual OHE
├── .gitignore
├── main.py                  # Entry point for tasks
├── README.md
└── requirements.txt
```

## Exploratory Tasks (4-9) Summary

### Task 4: Frequency Analysis
*   BoW captures frequency information by counting the occurrences of each word across the corpus. High-frequency words represent the common topics or themes, while low-frequency words may be specific to certain documents.

### Task 5 & 6: N-grams
*   As we increase the N-gram range (unigrams -> bigrams -> trigrams), the vocabulary size increases significantly due to the combinatorial growth of word sequences, providing more context at the cost of sparsity.

### Task 8: BoW vs TF-IDF
*   TF-IDF down-weights words that appear frequently across the entire corpus (high IDF) while highlighting words that are unique/discriminative to specific documents. BoW treats all frequent words equally, potentially allowing common words to dominate.

### Task 9: Hyperparameter Exploration
*   `max_features`: Limits the vocabulary to the top K most frequent words, reducing dimensionality.
*   `min_df`: Ignores words that appear in fewer than X documents, helping remove noise.
*   `max_df`: Ignores words that appear in more than X% of documents (e.g., stop words), focusing on unique content.

## Conceptual Questions (Task 10)
1. **OHE vs BoW:** OHE is binary (present/absent), while BoW is frequency-based.
2. **N-gram Dimensionality:** Increases combinatorially with sequences, leading to high dimensionality.
3. **TF-IDF over BoW:** Better for highlighting unique document-specific information vs global corpus statistics.
4. **Limitations of count-based:** Loss of semantic context and order; high sparsity.
