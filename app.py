# app.py
from flask import Flask, render_template, request
from model import load_books, build_model, recommend

app = Flask(__name__)
df = load_books()
similarity, df = build_model(df)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend_books():
    book = request.form['book']
    recommendations = recommend(book, df, similarity)
    return render_template('index.html', book=book, recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)
