# model.py
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_books():
    df = pd.read_csv('books.csv', encoding='latin-1', low_memory=False)

    # Keep only necessary columns
    df = df[['Book-Title', 'Book-Author']]

    # Remove empty rows and duplicates
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    # ✅ LIMIT to top 1000 rows to avoid memory issue
    df = df.head(1000)

    # Create a column combining title and author
    df['combined'] = df['Book-Title'] + " " + df['Book-Author']

    return df


def build_model(df):
    cv = CountVectorizer(stop_words='english')
    matrix = cv.fit_transform(df['combined'])
    similarity = cosine_similarity(matrix)
    return similarity, df

def recommend(book_name, df, similarity):
    if book_name not in df['Book-Title'].values:
        return ["Book not found"]
    idx = df[df['Book-Title'] == book_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    books = [df.iloc[i[0]]['Book-Title'] for i in scores]
    return books
