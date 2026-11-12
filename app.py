import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Setup NLTK
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# 1. Load & Understand Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    df['text'] = df['text'].fillna('')
    # Preprocessing
    df['clean_text'] = df['text'].str.lower().replace(r'[^\w\s]', '', regex=True)
    return df

df = load_data()

st.title("🎬 Recommendation System")

# Task 1: Displaying Dataset Info
if st.checkbox("Show Dataset Info"):
    st.write("### Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write(f"Columns: {df.columns.tolist()}")
    st.write("### First 5 rows:")
    st.dataframe(df.head())

# 2. Vectorization
# Using 'text' as the item identifier/content
tfidf = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Recommendation Logic
def recommend(item_text):
    # Find index using the 'text' column
    idx = df[df['text'] == item_text].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Top 5
    indices = [i[0] for i in sim_scores]
    return df['text'].iloc[indices]

# 4. Streamlit UI
# Use 'text' instead of 'title' as it is the available column
selected_item = st.selectbox("Select an item:", df['text'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_item)
    st.write("### Recommended similar items:")
    for item in recommendations:
        st.write(f"- {item}")
