API_BASE_URL = "https://pojokbaca-brida.my.id/api"

from flask import Flask, request, jsonify
import requests
import pandas as pd
import re
import nltk

from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from flask_cors import CORS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

  

app = Flask(__name__)
CORS(app, origins=[
    "https://pojokbaca-brida.my.id",
    "https://www.pojokbaca-brida.my.id"
])

# factory = StemmerFactory()
# stemmer = factory.create_stemmer()
 
nltk.download('stopwords', quiet=True)
stop_words_idn = set(stopwords.words('indonesian'))

factory = StemmerFactory()
stemmer = factory.create_stemmer()

@lru_cache(maxsize=50000)
def stem_cached(word):
    return stemmer.stem(word)

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Flask recommendation API is running"
    })
# =========================
# PREPROCESS
# =========================

# def preprocess(text):
#     # case folding
#     text = str(text).lower()

#     # punctuation removal / hapus simbol
#     text = re.sub(r'\\r\\n|\\n|\\r', ' ', text)
#     text = re.sub(r"[\'’‘`´]", '', text)
#     text = re.sub(r'[^\w\s]', ' ', text)
#     text = re.sub(r'\s+', ' ', text).strip()

#     # tokenization
#     tokens = text.split()

#     # stopword removal dan filtering kata
#     stopwords_removal = [
#         word for word in tokens
#         if word not in stop_words_idn and len(word) > 2
#     ]

#     # stemming
#     stemming = [
#         stemmer.stem(word)
#         for word in stopwords_removal
#     ]

#     return " ".join(stemming)
    
def preprocess(text):
    text = str(text).lower()

    text = re.sub(r'\\r\\n|\\n|\\r', ' ', text)
    text = re.sub(r"[\'’‘`´]", '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()

    tokens = [
        word for word in tokens
        if word not in stop_words_idn and len(word) > 2
    ]

    tokens = [
        stem_cached(word)
        for word in tokens
    ]

    return " ".join(tokens)


# =========================
# TABEL PREPROCESSING
# =========================
def generate_preprocessing_tables():
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    df = books.head(5).copy()

    df['Text'] = (
        df['title'].astype(str) + " " +
        df['subcategory'].astype(str) + " " +
        df['category'].astype(str) + " " +
        df['sinopsis'].astype(str)
    )

    # 1. Case Folding
    df['case_folding'] = df['Text'].astype(str).str.lower()

    # 2. Punctuation Removal
    def clean_punctuation_for_table(text):
        text = str(text)

        # hapus literal \r\n, \n, \r dari database
        text = re.sub(r'\\r\\n|\\n|\\r', ' ', text)

        # hapus apostrof tanpa spasi
        # contoh: ka'bah -> kabah
        text = re.sub(r"[\'’‘`´]", '', text)

        # hapus simbol lain
        text = re.sub(r'[^\w\s]', ' ', text)

        # rapikan spasi berlebih
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    df['punctuation_removal'] = df['case_folding'].apply(
        clean_punctuation_for_table
    )

    # 3. Tokenizing
    df['tokenizing'] = df['punctuation_removal'].apply(
        lambda x: x.split()
    )

    # 4. Stopword Removal
    df['stopword_removal'] = df['tokenizing'].apply(
        lambda tokens: [
            word for word in tokens
            if word not in stop_words_idn and len(word) > 2
        ]
    )

    # 5. Stemming
    # df['stemming'] = df['stopword_removal'].apply(
    #     lambda tokens: [
    #         stemmer.stem(word)
    #         for word in tokens
    #     ]
    # )

    df['stemming'] = df['stopword_removal'].apply(
        lambda tokens: [
            stem_cached(word)
            for word in tokens
        ]
    )

    return df


# =========================
# AMBIL DATA DARI PHP API
# =========================
def load_books():
    url = f"{API_BASE_URL}/books.php"
    # url = "https://domain.com/api/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    books['combined'] = (
        (books['title'].astype(str) + " ") * 4 +
        (books['subcategory'].astype(str) + " ") * 4 +
        (books['author'].astype(str) + " ") * 2 +
        (books['category'].astype(str) + " ") * 2 +
        (books['sinopsis'].astype(str) + " ") * 1
    )

    books['hasil'] = books['combined'].apply(preprocess)

    return books

books_cache = None
tfidf_cache = None
books_tfidf_cache = None

def prepare_model(force_refresh=False):
    global books_cache, tfidf_cache, books_tfidf_cache

    if (
        books_cache is not None
        and tfidf_cache is not None
        and books_tfidf_cache is not None
        and not force_refresh
    ):
        return books_cache, tfidf_cache, books_tfidf_cache

    books = load_books()

    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True
    )

    books_tfidf = tfidf.fit_transform(books['hasil'])

    books_cache = books
    tfidf_cache = tfidf
    books_tfidf_cache = books_tfidf

    return books_cache, tfidf_cache, books_tfidf_cache

@app.route('/refresh-model', methods=['GET'])
def refresh_model():
    prepare_model(force_refresh=True)

    return jsonify({
        "success": True,
        "message": "Model rekomendasi berhasil diperbarui."
    })
    
# =========================
# API REKOMENDASI
# =========================
# fetch("https://username.pythonanywhere.com/recommend", {
#   method: "POST",
#   headers: {
#     "Content-Type": "application/json"
#   },
#   body: JSON.stringify(payload)
# })

@app.route('/preprocessing-table', methods=['GET'])
def preprocessing_table():
    df = generate_preprocessing_tables()

    # Supaya kolom panjang tetap enak dibaca di browser
    pd.set_option('display.max_colwidth', 80)

    case_folding_table = df[['Text', 'case_folding']]
    punctuation_table = df[['case_folding', 'punctuation_removal']]
    tokenizing_table = df[['punctuation_removal', 'tokenizing']]
    stopword_table = df[['tokenizing', 'stopword_removal']]
    stemming_table = df[['stopword_removal', 'stemming']]
    # stemming_table = df[['stopword_removal', 'final_preprocessing']]

    return html


@app.route('/recommend', methods=['POST'])
def recommend():
    books, tfidf, books_tfidf = prepare_model()
    data = request.json

    # ambil preference user dari PHP API
    pref_response = requests.get(
        f"{API_BASE_URL}/get_preferences.php?username={data['username']}"
    )
    pref = pref_response.json()

    print("PREFERENCE:", pref)

    # ambil buku yang di-like user
    bookmarked_response = requests.post(
        f"{API_BASE_URL}/get_recent_favorites.php",
        json={
            "username": data['username']
        }
    )

    # ID buku yang di-bookmark / di-love user dari tabel favorite
    bookmarked_book_ids = bookmarked_response.json() or []

    # samakan tipe id buku agar cocok dengan id dari books.php
    books['id'] = books['id'].astype(str)
    bookmarked_book_ids = [
        str(book_id)
        for book_id in bookmarked_book_ids
    ]

    # TF-IDF
    # tfidf = TfidfVectorizer(
    #     ngram_range=(1, 2),
    #     min_df=2,
    #     sublinear_tf=True
    # )
    # books_tfidf = tfidf.fit_transform(books['hasil'])

    # buku yang di-favorite / bookmark user
    bookmarked_books = books[
        books['id'].isin(bookmarked_book_ids)
    ]

    # teks gabungan dari buku yang di-bookmark / di-love user
    bookmarked_text = ""

    for _, book in bookmarked_books.iterrows():
        bookmarked_text += " " + (
            str(book['title']) + " " +
            str(book['subcategory']) + " " +
            str(book['author']) + " " +
            str(book['category']) + " " +
            str(book['sinopsis'])
        )

    print("BOOKMARKED TEXT:", bookmarked_text)
    print("BOOK ID TYPES:", books['id'].head().tolist())
    print("BOOKMARKED BOOK IDS:", bookmarked_book_ids)
    print("BOOKMARKED BOOKS:", bookmarked_books[['id', 'title']].to_dict('records'))
    print("BOOKMARKED TEXT LENGTH:", len(bookmarked_text))

    # buku favorit yang dipilih user saat mengisi survey preferensi
    survey_favorite_books = pref.get('buku_favorit', [])

    # subkategori yang dipilih user saat mengisi survey preferensi
    preferred_subcategories = pref.get('sub_kategori', [])

    # kategori yang dipilih user saat mengisi survey preferensi
    preferred_categories = pref.get('kategori', [])

    # USER VECTOR
    user_text = (
        (" ".join(survey_favorite_books) + " ") * 4 +
        (bookmarked_text + " ") * 6 +
        (" ".join(preferred_subcategories) + " ") * 4 +
        (" ".join(preferred_categories) + " ") * 2
    )

    user_vec = tfidf.transform([
        preprocess(user_text)
    ])

    # similarity
    sim_scores = cosine_similarity(
        user_vec,
        books_tfidf
    ).flatten()

    # boost jika subkategori sama
    for idx in range(len(sim_scores)):
        book_subcategory = books.iloc[idx]['subcategory']

        if book_subcategory in preferred_subcategories:
            sim_scores[idx] += 0.20

    # buku yang sudah di-like / bookmark
    bookmarked_book_titles = bookmarked_books['title'].tolist()

    # gabungkan semua buku yang mau dikecualikan
    excluded_book_titles = [
        title.lower().strip()
        for title in (survey_favorite_books + bookmarked_book_titles)
    ]

    # urut similarity tertinggi
    top_idx = sim_scores.argsort()[::-1]

    results = []
    relevant_count = 0

    for i in top_idx:
        book_title = books.iloc[i]['title']
        book_subcategory = books.iloc[i]['subcategory']

        # skip buku yg sudah dipilih user
        if book_title.lower().strip() in excluded_book_titles:
            continue

        # cek relevansi berdasarkan subkategori user
        is_relevant = book_subcategory in preferred_subcategories

        if is_relevant:
            relevant_count += 1

        results.append({
            "id": books.iloc[i]['id'],
            "title": book_title,
            "author": books.iloc[i]['author'],
            "cover": books.iloc[i]['cover'],
            "subcategory": book_subcategory,
            "similarity": float(sim_scores[i]),
            "relevant": is_relevant
        })

        if len(results) >= 10:
            break

    precision = relevant_count / len(results) if len(results) > 0 else 0

    total_relevant = books[
        books['subcategory'].isin(preferred_subcategories)
    ].shape[0]

    recall = relevant_count / total_relevant if total_relevant > 0 else 0

    print("PRECISION@10:", precision)
    print("RECALL@10:", recall)

    return jsonify({
        "results": results,
        "evaluation": {
            "precision_at_10": precision,
            "recall_at_10": recall,
            "relevant_count": relevant_count,
            "total_recommended": len(results),
            "total_relevant_books": int(total_relevant)
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
