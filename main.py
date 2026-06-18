API_BASE_URL = "https://pojokbaca-brida.my.id/api"

from flask import Flask, request, jsonify
from flask_cors import CORS

import re
import nltk
import requests
import pandas as pd

from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# INISIALISASI APLIKASI
# =========================
app = Flask(__name__)

CORS(app, origins=[
    "https://pojokbaca-brida.my.id",
    "https://www.pojokbaca-brida.my.id"
])

nltk.download("stopwords", quiet=True)

stop_words_idn = set(stopwords.words("indonesian"))

factory = StemmerFactory()
stemmer = factory.create_stemmer()
#


# =========================
# STEMMING CACHE
# =========================
@lru_cache(maxsize=50000)
def stem_cached(word):
    return stemmer.stem(word)
#


# =========================
# ROUTE UTAMA
# =========================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Flask recommendation API is running"
    })
#


# =========================
# PREPROCESSING
# =========================
def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"\\r\\n|\\n|\\r", " ", text)
    text = re.sub(r"[\'’‘`´]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

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
#


# =========================
# TABEL PREPROCESSING
# =========================
def generate_preprocessing_tables():
    # data books
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    df = books.head(5).copy()

    df["Text"] = (
        df["title"].astype(str) + " " +
        df["subcategory"].astype(str) + " " +
        df["author"].astype(str) + " " +
        df["category"].astype(str) + " " +
        df["sinopsis"].astype(str)
    )

    df["case_folding"] = df["Text"].astype(str).str.lower()

    df["punctuation_removal"] = df["case_folding"].apply(clean_text_for_table)

    df["tokenizing"] = df["punctuation_removal"].apply(
        lambda text: text.split()
    )

    df["stopword_removal"] = df["tokenizing"].apply(
        lambda tokens: [
            word for word in tokens
            if word not in stop_words_idn and len(word) > 2
        ]
    )

    df["stemming"] = df["stopword_removal"].apply(
        lambda tokens: [
            stem_cached(word)
            for word in tokens
        ]
    )

    return df


def clean_text_for_table(text):
    text = str(text)

    text = re.sub(r"\\r\\n|\\n|\\r", " ", text)
    text = re.sub(r"[\'’‘`´]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
#
@app.route("/preprocessing-table", methods=["GET"])
def preprocessing_table():
    try:
        df = generate_preprocessing_tables()

        selected_columns = [
            "title",
            "Text",
            "case_folding",
            "punctuation_removal",
            "tokenizing",
            "stopword_removal",
            "stemming"
        ]

        df = df[
            [col for col in selected_columns if col in df.columns]
        ].copy()

        # Biar list token enak dibaca di tabel HTML
        for col in ["tokenizing", "stopword_removal", "stemming"]:
            if col in df.columns:
                df[col] = df[col].apply(
                    lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                )

        table_html = df.to_html(
            index=False,
            escape=True,
            classes="preprocessing-table"
        )

        return f"""
        <!DOCTYPE html>
        <html lang="id">
        <head>
            <meta charset="UTF-8">
            <title>Tabel Preprocessing</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 24px;
                    background: #f8fafc;
                    color: #1e293b;
                }}

                h1 {{
                    font-size: 24px;
                    margin-bottom: 16px;
                }}

                .table-wrapper {{
                    overflow-x: auto;
                    background: white;
                    padding: 16px;
                    border-radius: 12px;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
                }}

                table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-size: 13px;
                }}

                th {{
                    background: #112D4E;
                    color: white;
                    padding: 10px;
                    text-align: left;
                    white-space: nowrap;
                }}

                td {{
                    border: 1px solid #e2e8f0;
                    padding: 10px;
                    vertical-align: top;
                    min-width: 180px;
                    max-width: 420px;
                }}

                tr:nth-child(even) {{
                    background: #f8fafc;
                }}
            </style>
        </head>
        <body>
            <h1>Tabel Hasil Preprocessing</h1>
            <p>Data berikut menampilkan tahapan case folding, punctuation removal, tokenizing, stopword removal, dan stemming.</p>

            <div class="table-wrapper">
                {table_html}
            </div>
        </body>
        </html>
        """

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500

# =========================
# PEMBENTUKAN TEKS BUKU
# =========================
def build_book_text(book, include_sinopsis=True):
    text = (
        str(book.get("title", "")) + " " +
        str(book.get("subcategory", "")) + " " +
        str(book.get("author", "")) + " " +
        str(book.get("category", ""))
    )

    if include_sinopsis:
        text += " " + str(book.get("sinopsis", ""))

    return text
#

# 
def clean_list(values):
    if values is None:
        return []

    if isinstance(values, list):
        raw_values = values
    else:
        raw_values = [values]

    return [
        str(value).strip()
        for value in raw_values
        if str(value).strip()
    ]
# 

# =========================
# LOAD DATA BUKU
# =========================
def load_books():
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    books["combined"] = (
        (books["subcategory"].astype(str) + " ") * 4 +
        (books["title"].astype(str) + " ") * 3 +
        (books["author"].astype(str) + " ") * 2 +
        (books["category"].astype(str) + " ") * 1 + 
        (books["sinopsis"].astype(str) + " ") * 1
    )

    books["hasil"] = books["combined"].apply(preprocess)

    return books
#


# =========================
# MODEL CACHE
# =========================
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

    books_tfidf = tfidf.fit_transform(books["hasil"])

    books_cache = books
    tfidf_cache = tfidf
    books_tfidf_cache = books_tfidf

    return books_cache, tfidf_cache, books_tfidf_cache


@app.route("/refresh-model", methods=["GET"])
def refresh_model():
    prepare_model(force_refresh=True)

    return jsonify({
        "success": True,
        "message": "Model rekomendasi berhasil diperbarui."
    })
#


# =========================
# PREFERENSI USER
# =========================
def get_user_preferences(username):
    response = requests.get(
        f"{API_BASE_URL}/get_preferences.php?username={username}"
    )

    return response.json()

# data bookmark
def get_recent_bookmarks(username):
    response = requests.post(
        f"{API_BASE_URL}/get_recent_favorites.php",
        json={
            "username": username
        }
    )

    bookmarked_book_ids = response.json() or []

    return [
        str(book_id)
        for book_id in bookmarked_book_ids
    ]
    # (judul, subk, penulis, kateg, sinopsis)

# normalisasi buku favorit/bookmark
def normalize_title(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*:\s*", ":", text)
    return text

def get_books_by_titles(books, titles):
    normalized_titles = [
        normalize_title(title)
        for title in titles
    ]

    return books[
        books["title"].apply(normalize_title).isin(normalized_titles)
    ]


def build_books_text(books_df, include_sinopsis=True):
    text = ""

    for _, book in books_df.iterrows():
        text += " " + build_book_text(
            book,
            include_sinopsis=include_sinopsis
        )

    return text
#


# =========================
# REKOMENDASI
# =========================
@app.route("/recommend", methods=["POST", "OPTIONS"])
def recommend():
    if request.method == "OPTIONS":
        return jsonify({
            "success": True
        }), 200

    data = request.json or {}
    username = data.get("username")

    if not username:
        return jsonify({
            "success": False,
            "message": "Username tidak ditemukan."
        }), 400

    books, tfidf, books_tfidf = prepare_model()

    books["id"] = books["id"].astype(str)

    pref = get_user_preferences(username)

    survey_favorite_books = clean_list(
        pref.get("buku_favorit", [])
    )

    preferred_subcategories = clean_list(
        pref.get("sub_kategori", [])
    )

    preferred_categories = clean_list(
        pref.get("kategori", [])
    )

    # =========================
    # DATA BOOKMARK TERBARU USER
    # =========================
    bookmarked_book_ids = get_recent_bookmarks(username)

    bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids)
    ]

    # =========================
    # DATA BUKU FAVORIT AWAL USER
    # =========================
    survey_favorite_books_df = get_books_by_titles(
        books,
        survey_favorite_books
    )

    # =========================
    # PEMBENTUKAN GROUND TRUTH / PROXY RELEVANCE
    # =========================
    survey_favorite_subcategories = clean_list(
        survey_favorite_books_df["subcategory"].dropna().tolist()
    )

    bookmarked_subcategories = clean_list(
        bookmarked_books["subcategory"].dropna().tolist()
    )

    relevance_subcategories = sorted(set(
        preferred_subcategories +
        survey_favorite_subcategories 
        # bookmarked_subcategories
    ))

    # =========================
    # PEMBENTUKAN PROFIL USER
    # =========================
    bookmarked_text = build_books_text(
        bookmarked_books,
        include_sinopsis=False
    )

    survey_favorite_text = build_books_text(
        survey_favorite_books_df,
        include_sinopsis=False
    )

    user_text = (
        (" ".join(preferred_subcategories) + " ") * 4 +
        
        #judul, penulis, kategori, sub kategori
        (survey_favorite_text + " ") * 2 + 
        
        # judul, penulis, kategori, sub kategori
        (bookmarked_text + " ") * 1 +
        
        (" ".join(preferred_categories) + " ") * 1
    )

    user_vec = tfidf.transform([
        preprocess(user_text)
    ])

    sim_scores = cosine_similarity(
        user_vec,
        books_tfidf
    ).flatten()

    # =========================
    # MENGECUALIKAN BUKU YANG SUDAH JADI PREFERENSI / BOOKMARK
    # =========================
    bookmarked_book_titles = bookmarked_books["title"].tolist()

    excluded_book_titles = [
        normalize_title(title)
        for title in (survey_favorite_books + bookmarked_book_titles)
    ]

    top_idx = sim_scores.argsort()[::-1]

    results = []
    relevant_count = 0

    for i in top_idx:
        book = books.iloc[i]

        book_title = str(book["title"]).strip()
        book_subcategory = str(book["subcategory"]).strip()

        if normalize_title(book_title) in excluded_book_titles:
            continue

        is_relevant = book_subcategory in relevance_subcategories

        if is_relevant:
            relevant_count += 1

        results.append({
            "id": book["id"],
            "title": book_title,
            "author": book["author"],
            "cover": book["cover"],
            "subcategory": book_subcategory,
            "similarity": float(sim_scores[i]),
            "relevant": is_relevant
        })

        if len(results) >= 10:
            break

    total_recommended = len(results)

    precision = (
        relevant_count / total_recommended
        if total_recommended > 0
        else 0
    )

    # =========================
    # TOTAL BUKU RELEVAN DI DATABASE
    # buku yang sudah masuk preferensi/bookmark tidak dihitung
    # karena memang dikecualikan dari hasil rekomendasi
    # =========================
    candidate_books_for_eval = books[
        ~books["title"].apply(normalize_title).isin(excluded_book_titles)
    ].copy()

    total_relevant = candidate_books_for_eval[
        candidate_books_for_eval["subcategory"]
        .astype(str)
        .str.strip()
        .isin(relevance_subcategories)
    ].shape[0]

    recall = (
        relevant_count / total_relevant
        if total_relevant > 0
        else 0
    )

    return jsonify({
        "results": results,
        "evaluation": {
            "precision_at_10": precision,
            "recall_at_10": recall,
            "relevant_count": relevant_count,
            "total_recommended": total_recommended,
            "total_relevant_books": int(total_relevant),
            "relevance_subcategories": relevance_subcategories
        }
    })
#


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
#
