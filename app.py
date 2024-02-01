import streamlit as st 
import pandas as pd
import numpy as np
import re
# from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import plotly.io as pio


# Path: data.py
df = pd.read_csv('/Users/mac/Desktop/research_paper_recommendation/cleaned_data.csv', dtype={'paper_id': 'str'})

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['stemmed_sentence'])

tfidf_vectorizer2 = TfidfVectorizer(max_features=10000)
tfidf_matrix2 = tfidf_vectorizer.fit_transform(df['stemmed_sentence'])

def get_recommendations(value, tfidf_vectorizer, tfidf_matrix, df, num_recommendations):
    if value in df['paper_id'].values:
        paper_id = value
        keywords = df[df['paper_id'] == paper_id]['keywords'].values[0]
    else:
        keywords = value.lower()
        keywords = re.sub(r"[^\w\s]", "", keywords)

    keyword_vector = tfidf_vectorizer.transform([keywords]).toarray()
    keyword_vector = keyword_vector[:, :tfidf_matrix.shape[1]]
    similarities = cosine_similarity(keyword_vector, tfidf_matrix)
    top_n_idx = np.argsort(-similarities, axis=1)[:, :num_recommendations].flatten()
    rec_paper_id_title = df.loc[top_n_idx, ['paper_id', 'title']].values.tolist()
    return rec_paper_id_title

def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
    
    keywords.sort(key=lambda x: x[1])
    keywords.reverse()
    return_values = []
    for keyword in keywords:
        return_values.append(keyword[0])
    return " ".join(return_values)


def search_dataframe(df, column_name, keywords, num_results=None):
    if isinstance(keywords, str):
        keywords = [keywords]
    
    pattern = "|".join([re.escape(keyword) for keyword in keywords])
    matches = df[column_name].str.findall(pattern, flags=re.IGNORECASE)
    mask = matches.apply(lambda x: len(x) > 0)
    result = df[mask]

    if num_results is not None: 
        result = result.head(num_results)
        result = result[["paper_id", "title"]]
    else:
        return "Oops, We can't find articles related to this"
    

    return result

def main():
    st.title('Computer Science Article Recommendation System')

    # sidebar
    page = st.sidebar.selectbox("Choose a page", ['Home', 'Search by Name', "Get Recommendations", "Search by ID"])

    if page == 'Home':
        st.header('Welcome to the Recommendation System')
        st.text('Use the sidebar to navigate to different pages.')

    elif page == 'Search by Name':
        st.header('Search by Name')
        rnam = st.text_input('Enter the name of the research paper: ')
        npapers = st.number_input('Number of recommendations to display:', min_value=1, step=1, value=5)
        
        if st.button('Search'):
            result= search_dataframe(df, 'title', rnam, npapers)
            st.table(result)
    
            
    elif page == 'Get Recommendations':
        st.header('Get Recommedations')
        rid= st.text_input('Enter Keyword: ')
        npapers = st.number_input('Number of recommendations to display:', min_value=1, step=1, value=5)
        if st.button('Get Recommendations'):
            recommendations = get_recommendations(rid, tfidf_vectorizer, tfidf_matrix,df, npapers)
            if recommendations:
                st.write('Top 5 Recommended Articles:')
                st.table(recommendations)
            else:
                st.write('No recommendations found.')


    

    elif page == 'Search by ID':
        st.header('Search by ID')
        rid = st.text_input('Enter the ID of the research paper: ')

        if st.button('Search by ID'):
            ans = df[df['paper_id'] == rid]
            ans = ans.to_dict('records')
            st.table(ans)
            



if __name__== "__main__":
    main()