API_BASE_URL = "https://pojokbaca-brida.my.id/api"

MAX_BOOKMARKS_FOR_RECOMMENDATION  = 1

from flask import Flask, request, jsonify
from flask_cors import CORS

import re
import json
import nltk
import requests
import pandas as pd

from functools import lru_cache
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# SETUP APLIKASI
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


def clean_text_for_table(text):
    text = str(text)
    text = re.sub(r"\\r\\n|\\n|\\r", " ", text)
    text = re.sub(r"[\'’‘`´]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_preprocessing_steps(text):
    case_folding = str(text).lower()
    punctuation_removal = clean_text_for_table(case_folding)
    tokenizing = punctuation_removal.split()

    stopword_removal = [
        word for word in tokenizing
        if word not in stop_words_idn and len(word) > 2
    ]

    stemming = [
        stem_cached(word)
        for word in stopword_removal
    ]

    return {
        "Case Folding": case_folding,
        "Punctuation Removal": punctuation_removal,
        "Tokenizing": tokenizing,
        "Stopword Removal": stopword_removal,
        "Stemming": stemming
    }


# =========================
# PARSING DATA JSON DARI DATABASE
# =========================
def parse_preference_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        raw_values = value

    elif isinstance(value, str):
        text = value.strip()

        if not text:
            return []

        try:
            parsed = json.loads(text)

            if isinstance(parsed, list):
                raw_values = parsed
            else:
                raw_values = [parsed]

        except Exception:
            raw_values = [text]

    else:
        raw_values = [value]

    result = []

    for item in raw_values:
        item = str(item).strip()

        if item:
            result.append(item)

    return result


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


def build_books_text(books_df, include_sinopsis=True):
    text = ""

    for _, book in books_df.iterrows():
        text += " " + build_book_text(
            book,
            include_sinopsis=include_sinopsis
        )

    return text


# =========================
# LOAD DATA BUKU
# =========================
def load_books():
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    # Pembobotan atribut buku
    books["combined"] = (
        (books["subcategory"].astype(str) + " ") +
        (books["title"].astype(str) + " ") +
        (books["author"].astype(str) + " ") +
        (books["category"].astype(str) + " ")  +
        (books["sinopsis"].astype(str) + " ")
    )

    books["hasil"] = books["combined"].apply(preprocess)

    return books


# =========================
# MODEL TF-IDF
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


# =========================
# API DATA USER
# =========================
def get_user_preferences(username):
    response = requests.get(
        f"{API_BASE_URL}/get_preferences.php?username={username}"
    )

    if not response.ok:
        return {}

    pref = response.json()

    if not isinstance(pref, dict):
        return {}

    return pref


def get_recent_bookmarks(username):
    response = requests.post(
        f"{API_BASE_URL}/get_recent_favorites.php",
        json={
            "username": username
        }
    )

    if not response.ok:
        return []

    bookmarked_book_ids = response.json() or []

    return [
        str(book_id)
        for book_id in bookmarked_book_ids
    ]


# =========================
# TABEL PREPROCESSING
# =========================
def generate_preprocessing_tables():
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    books["id"] = books["id"].astype(str)

    target_book = books[books["id"] == "34"].copy()
    other_books = books[books["id"] != "34"].head(5).copy()

    df = pd.concat([target_book, other_books], ignore_index=True)

    df["merged_text"] = (
        df["title"].astype(str) + " " +
        df["subcategory"].astype(str) + " " +
        df["author"].astype(str) + " " +
        df["category"].astype(str) + " " +
        df["sinopsis"].astype(str)
    )

    # Pembobotan atribut buku
    df["weighted_text"] = (
        (df["subcategory"].astype(str) + " ") +
        (df["title"].astype(str) + " ") +
        (df["author"].astype(str) + " ") +
        (df["category"].astype(str) + " ")+
        (df["sinopsis"].astype(str) + " ") 
    )

    preprocessing_rows = []

    for _, row in df.iterrows():
        steps = get_preprocessing_steps(row["weighted_text"])

        preprocessing_rows.append({
            "Jenis Dokumen": "Buku",
            "ID Buku": row.get("id", ""),
            "Judul Buku": row.get("title", ""),
            "Subkategori": row.get("subcategory", ""),
            "Kategori": row.get("category", ""),
            "Penulis": row.get("author", ""),
            "Dokumen Hasil Merge Data": row["merged_text"],
            "Dokumen Hasil Pembobotan Atribut": row["weighted_text"],
            **steps
        })

    return pd.DataFrame(preprocessing_rows), books


def generate_user_preprocessing_table(username, books):
    if not username:
        return pd.DataFrame()

    books = books.copy()
    books["id"] = books["id"].astype(str)

    pref = get_user_preferences(username)

    # Data dari form preferensi
    preference_selected_books = parse_preference_list(
        pref.get("buku_pilihan", pref.get("buku_favorit", []))
    )

    preferred_subcategories = parse_preference_list(
        pref.get("sub_kategori", [])
    )

    preferred_categories = parse_preference_list(
        pref.get("kategori", [])
    )

    # Data favorit katalog/bookmark
    bookmarked_book_ids = get_recent_bookmarks(username)

    bookmarked_book_ids_for_recommendation = bookmarked_book_ids[
        :MAX_BOOKMARKS_FOR_RECOMMENDATION
    ]
    
    bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids_for_recommendation)
    ]
    
    selected_books_df = get_books_by_titles(
        books,
        preference_selected_books
    )

    selected_book_text = build_books_text(
        selected_books_df,
        include_sinopsis=False
    )

    bookmarked_text = build_books_text(
        bookmarked_books,
        include_sinopsis=False
    )

    merged_user_text = (
        " ".join(preferred_subcategories) + " " +
        selected_book_text + " " +
        bookmarked_text + " " +
        " ".join(preferred_categories)
    )

    # Pembobotan preferensi pengguna
    weighted_user_text = (
        (" ".join(preferred_subcategories) + " ")+
        (selected_book_text + " ") +
        (bookmarked_text + " ") +
        (" ".join(preferred_categories) + " ") 
    )

    steps = get_preprocessing_steps(weighted_user_text)

    selected_book_subcategories = (
        selected_books_df["subcategory"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if not selected_books_df.empty else []
    )

    bookmarked_titles = (
        bookmarked_books["title"]
        .dropna()
        .astype(str)
        .tolist()
        if not bookmarked_books.empty else []
    )

    bookmarked_subcategories = (
        bookmarked_books["subcategory"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
        if not bookmarked_books.empty else []
    )

    user_row = {
        "Jenis Dokumen": "Preferensi Pengguna",
        "Username": username,

        "Subkategori Pilihan": ", ".join(preferred_subcategories),
        "Kategori Pilihan": ", ".join(preferred_categories),

        "Buku Pilihan": ", ".join(preference_selected_books),
        "Subkategori Buku Pilihan": ", ".join(selected_book_subcategories),

        "Buku Favorit Katalog": ", ".join(bookmarked_titles),
        "Subkategori Buku Favorit Katalog": ", ".join(bookmarked_subcategories),

        "Dokumen Hasil Merge Data": merged_user_text,
        "Dokumen Hasil Pembobotan Atribut": weighted_user_text,

        **steps
    }

    return pd.DataFrame([user_row])


@app.route("/preprocessing-table", methods=["GET"])
def preprocessing_table():
    try:
        username = request.args.get("username", "").strip()

        book_df, books = generate_preprocessing_tables()

        user_df = generate_user_preprocessing_table(
            username,
            books
        )

        for df in [book_df, user_df]:
            for col in ["Tokenizing", "Stopword Removal", "Stemming"]:
                if col in df.columns:
                    df[col] = df[col].apply(
                        lambda x: ", ".join(x) if isinstance(x, list) else str(x)
                    )

        book_table_html = book_df.to_html(
            index=False,
            escape=True,
            classes="preprocessing-table"
        )

        if not user_df.empty:
            user_table_html = user_df.to_html(
                index=False,
                escape=True,
                classes="preprocessing-table"
            )
        else:
            user_table_html = """
            <p class="note">
                Data preprocessing pengguna belum ditampilkan karena parameter username belum diberikan.
                Contoh akses: <b>/preprocessing-table?username=titi</b>
            </p>
            """

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
                    margin-bottom: 8px;
                }}

                h2 {{
                    font-size: 20px;
                    margin-top: 32px;
                    margin-bottom: 12px;
                }}

                p {{
                    line-height: 1.6;
                }}

                .note {{
                    background: #fff7ed;
                    border: 1px solid #fed7aa;
                    padding: 12px;
                    border-radius: 10px;
                    color: #9a3412;
                }}

                .table-wrapper {{
                    overflow-x: auto;
                    background: white;
                    padding: 16px;
                    border-radius: 12px;
                    box-shadow: 0 8px 24px rgba(0,0,0,0.08);
                    margin-bottom: 28px;
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
            <p>
                Data berikut menampilkan tahapan preprocessing setelah dokumen melalui
                proses merge data dan pembobotan atribut.
            </p>

            <h2>1. Preprocessing Dokumen Buku</h2>
            <div class="table-wrapper">
                {book_table_html}
            </div>

            <h2>2. Preprocessing Dokumen Preferensi Pengguna</h2>
            <div class="table-wrapper">
                {user_table_html}
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

    # Data dari form preferensi
    preference_selected_books = parse_preference_list(
        pref.get("buku_pilihan", pref.get("buku_favorit", []))
    )
 
    preferred_subcategories = parse_preference_list(
        pref.get("sub_kategori", [])
    )

    preferred_categories = parse_preference_list(
        pref.get("kategori", [])
    )

    # Data favorit katalog/bookmark
    bookmarked_book_ids = get_recent_bookmarks(username)
    
    bookmarked_book_ids_for_recommendation = bookmarked_book_ids[
        :MAX_BOOKMARKS_FOR_RECOMMENDATION
    ]
    
    # Buku favorit katalog yang masuk perhitungan rekomendasi, maksimal 1
    bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids_for_recommendation)
    ]
    
    # Buku pilihan form tetap dipakai semua
    selected_books_df = get_books_by_titles(
        books,
        preference_selected_books
    )

    # Acuan relevansi evaluasi
    selected_book_subcategories = parse_preference_list(
        selected_books_df["subcategory"].dropna().tolist()
    )

    relevance_subcategories = sorted(set(
        preferred_subcategories +
        selected_book_subcategories
    ))

    # Pembentukan profil pengguna
    selected_book_text = build_books_text(
        selected_books_df,
        include_sinopsis=False
    )

    bookmarked_text = build_books_text(
        bookmarked_books,
        include_sinopsis=False
    )

    user_text = (
        (" ".join(preferred_subcategories) + " ")+
        (selected_book_text + " ")  +
        (bookmarked_text + " ")  +
        (" ".join(preferred_categories) + " ")
    )

    user_vec = tfidf.transform([
        preprocess(user_text)
    ])

    sim_scores = cosine_similarity(
        user_vec,
        books_tfidf
    ).flatten()

    # Buku pilihan dan semua bookmark tidak ditampilkan ulang
    all_bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids)
    ]
    
    bookmarked_book_titles = all_bookmarked_books["title"].tolist()
    
    excluded_book_titles = [
        normalize_title(title)
        for title in (preference_selected_books + bookmarked_book_titles)
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


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(debug=True)
