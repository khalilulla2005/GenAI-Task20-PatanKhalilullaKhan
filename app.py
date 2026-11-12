import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup NLTK
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load & Understand Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    df['text'] = df['text'].fillna('')
    return df

df = load_data()

st.title("🎬 Movie Recommendation System")

# Task 1: Displaying Dataset Info
if st.checkbox("Show Dataset Info"):
    st.write("### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write("### First 5 rows:")
    st.dataframe(df.head())

# Task 2: Text Preprocessing
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = text.replace(r'[^\w\s]', '')
    # Stopwords
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_text'] = df['text'].apply(preprocess_text)

# 3. Vectorization
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 4. Recommendation Logic
def recommend(movie_title):
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# 5. Streamlit UI
selected_item = st.selectbox("Select a movie:", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_item)
    st.write("### Recommended for you:")
    for item in recommendations:
        st.write(f"- {item}")
