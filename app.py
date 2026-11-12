import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('data/dataset.csv')
    # Assuming 'text' is the column with descriptions
    df['text'] = df['text'].fillna('')
    # Preprocessing
    df['clean_text'] = df['text'].str.lower().replace(r'[^\w\s]', '', regex=True)
    return df

df = load_data()

# 2. Vectorization
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_matrix = tfidf.fit_transform(df['clean_text'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Recommendation Function
def recommend(movie_title):
    # This logic assumes 'title' or equivalent identifier exists in dataset
    # You may need to adjust the column name based on your CSV
    idx = df[df['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6] # Top 5
    movie_indices = [i[0] for i in sim_scores]
    return df['title'].iloc[movie_indices]

# 4. Streamlit UI
st.title("🎬 Item Recommendation System")
selected_item = st.selectbox("Select an item:", df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_item)
    st.write("### Recommended for you:")
    for item in recommendations:
        st.write(f"- {item}")
