API_BASE_URL = "https://pojokbaca-brida.my.id/api"
MAX_BOOKMARKS_FOR_RECOMMENDATION = 1

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


# =========================================================
# KONFIGURASI APLIKASI
# =========================================================
app = Flask(__name__)

CORS(app, origins=[
    "https://pojokbaca-brida.my.id",
    "https://www.pojokbaca-brida.my.id"
])

nltk.download("stopwords", quiet=True)

stop_words_idn = set(stopwords.words("indonesian"))

factory = StemmerFactory()
stemmer = factory.create_stemmer()


FIELD_COLUMNS = [
    "subcategory",
    "title",
    "author",
    "category",
    "sinopsis"
]

FIELD_WEIGHTS = {
    "subcategory": 2,
    "title": 2,
    "author": 1,
    "category": 1,
    "sinopsis": 1
}


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "Flask recommendation API is running"
    })


# =========================================================
# PREPROCESSING
# =========================================================
@lru_cache(maxsize=50000)
def stem_cached(word):
    return stemmer.stem(word)


def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"\\r\\n|\\n|\\r", " ", text)
    text = re.sub(r"[\'’‘`´]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()

    tokens = [
        word
        for word in tokens
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
        word
        for word in tokenizing
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


# =========================================================
# HELPER DATA
# =========================================================
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

    return [
        str(item).strip()
        for item in raw_values
        if str(item).strip()
    ]


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
        books["title"]
        .apply(normalize_title)
        .isin(normalized_titles)
    ]


# =========================================================
# DOKUMEN BUKU
# =========================================================
def load_book_documents():
    response = requests.get(
        f"{API_BASE_URL}/books.php"
    )

    books = pd.DataFrame(response.json())
    books["id"] = books["id"].astype(str)

    # Setiap field buku menjadi dokumen tersendiri.
    for field in FIELD_COLUMNS:
        books[f"{field}_clean"] = (
            books[field]
            .fillna("")
            .astype(str)
            .apply(preprocess)
        )

    # Hanya untuk penyajian dokumen gabungan di laporan.
    books["merged_clean"] = books.apply(
        lambda row: " ".join(
            row[f"{field}_clean"]
            for field in FIELD_COLUMNS
            if row[f"{field}_clean"]
        ),
        axis=1
    )

    return books


# =========================================================
# TF-IDF DOKUMEN BUKU
# =========================================================
books_cache = None
tfidf_cache = None
books_tfidf_cache = None


def prepare_model(force_refresh=False):
    global books_cache
    global tfidf_cache
    global books_tfidf_cache

    if (
        books_cache is not None
        and tfidf_cache is not None
        and books_tfidf_cache is not None
        and not force_refresh
    ):
        return (
            books_cache,
            tfidf_cache,
            books_tfidf_cache
        )

    books = load_book_documents()

    tfidf_cache = {}
    books_tfidf_cache = {}

    for field in FIELD_COLUMNS:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            books[f"{field}_clean"]
        )

        tfidf_cache[field] = vectorizer
        books_tfidf_cache[field] = matrix

    books_cache = books

    return (
        books_cache,
        tfidf_cache,
        books_tfidf_cache
    )


@app.route("/refresh-model", methods=["GET"])
def refresh_model():
    prepare_model(force_refresh=True)

    return jsonify({
        "success": True,
        "message": "Model rekomendasi berhasil diperbarui."
    })


# =========================================================
# DATA PREFERENSI PENGGUNA
# =========================================================
def get_user_preferences(username):
    response = requests.get(
        f"{API_BASE_URL}/get_preferences.php",
        params={"username": username}
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
        json={"username": username}
    )

    if not response.ok:
        return []

    bookmarked_book_ids = response.json() or []

    return [
        str(book_id)
        for book_id in bookmarked_book_ids
    ]


# =========================================================
# DOKUMEN PROFIL PENGGUNA
# =========================================================
def build_user_field_texts(
    pref,
    selected_books_df,
    bookmarked_books
):
    preferred_subcategories = parse_preference_list(
        pref.get("sub_kategori", [])
    )

    preferred_categories = parse_preference_list(
        pref.get("kategori", [])
    )

    return {
        "subcategory": " ".join(
            preferred_subcategories
            + selected_books_df[
                "subcategory"
            ].astype(str).tolist()
            + bookmarked_books[
                "subcategory"
            ].astype(str).tolist()
        ),

        "title": " ".join(
            selected_books_df[
                "title"
            ].astype(str).tolist()
            + bookmarked_books[
                "title"
            ].astype(str).tolist()
        ),

        "author": " ".join(
            selected_books_df[
                "author"
            ].astype(str).tolist()
            + bookmarked_books[
                "author"
            ].astype(str).tolist()
        ),

        "category": " ".join(
            preferred_categories
            + selected_books_df[
                "category"
            ].astype(str).tolist()
            + bookmarked_books[
                "category"
            ].astype(str).tolist()
        ),

        "sinopsis": ""
    }


def merge_user_text_for_display(user_field_texts):
    return " ".join(
        preprocess(user_field_texts.get(field, ""))
        for field in FIELD_COLUMNS
        if user_field_texts.get(field, "").strip()
    )


# =========================================================
# WEIGHTED COSINE SIMILARITY
# =========================================================
def weighted_cosine_similarity(
    user_field_texts,
    n_books,
    tfidf_dict,
    books_tfidf_dict,
    weights
):
    total_score = None
    total_weight = 0

    for field, weight in weights.items():
        user_text = user_field_texts.get(
            field,
            ""
        )

        if not user_text.strip():
            continue

        vectorizer = tfidf_dict[field]
        books_matrix = books_tfidf_dict[field]

        user_vector = vectorizer.transform([
            preprocess(user_text)
        ])

        field_similarity = cosine_similarity(
            user_vector,
            books_matrix
        ).flatten()

        if total_score is None:
            total_score = (
                weight * field_similarity
            )
        else:
            total_score += (
                weight * field_similarity
            )

        total_weight += weight

    if total_score is None or total_weight == 0:
        return [0.0] * n_books

    return total_score / total_weight


# =========================================================
# DATA UNTUK TABEL PREPROCESSING
# =========================================================
def generate_book_preprocessing_table(books):
    target_book = books[
        books["id"] == "34"
    ].copy()

    other_books = books[
        books["id"] != "34"
    ].head(5).copy()

    sample_books = pd.concat(
        [target_book, other_books],
        ignore_index=True
    )

    rows = []

    for _, book in sample_books.iterrows():
        for field in FIELD_COLUMNS:
            steps = get_preprocessing_steps(
                book.get(field, "")
            )

            rows.append({
                "Jenis Dokumen": "Buku",
                "ID Buku": book.get("id", ""),
                "Judul Buku": book.get("title", ""),
                "Field": field,
                "Teks Asli": book.get(field, ""),
                **steps
            })

    return pd.DataFrame(rows)


def generate_user_preprocessing_table(
    username,
    books
):
    if not username:
        return pd.DataFrame()

    pref = get_user_preferences(username)

    selected_titles = parse_preference_list(
        pref.get(
            "buku_pilihan",
            pref.get("buku_favorit", [])
        )
    )

    selected_books = get_books_by_titles(
        books,
        selected_titles
    )

    bookmarked_ids = get_recent_bookmarks(
        username
    )

    bookmarked_ids_for_profile = (
        bookmarked_ids[
            :MAX_BOOKMARKS_FOR_RECOMMENDATION
        ]
    )

    bookmarked_books = books[
        books["id"].isin(
            bookmarked_ids_for_profile
        )
    ]

    user_field_texts = build_user_field_texts(
        pref,
        selected_books,
        bookmarked_books
    )

    rows = []

    for field in FIELD_COLUMNS:
        text = user_field_texts.get(
            field,
            ""
        )

        steps = get_preprocessing_steps(
            text
        )

        rows.append({
            "Jenis Dokumen":
                "Preferensi Pengguna",
            "Username":
                username,
            "Field":
                field,
            "Teks Asli":
                text,
            "Bobot":
                FIELD_WEIGHTS[field],
            **steps
        })

    return pd.DataFrame(rows)


@app.route(
    "/preprocessing-table",
    methods=["GET"]
)
def preprocessing_table():
    try:
        username = request.args.get(
            "username",
            ""
        ).strip()

        books = load_book_documents()

        book_df = (
            generate_book_preprocessing_table(
                books
            )
        )

        user_df = (
            generate_user_preprocessing_table(
                username,
                books
            )
        )

        for df in [book_df, user_df]:
            for column in [
                "Tokenizing",
                "Stopword Removal",
                "Stemming"
            ]:
                if column in df.columns:
                    df[column] = df[
                        column
                    ].apply(
                        lambda value:
                        ", ".join(value)
                        if isinstance(
                            value,
                            list
                        )
                        else str(value)
                    )

        book_table_html = book_df.to_html(
            index=False,
            escape=True,
            classes="preprocessing-table"
        )

        if not user_df.empty:
            user_table_html = (
                user_df.to_html(
                    index=False,
                    escape=True,
                    classes="preprocessing-table"
                )
            )
        else:
            user_table_html = """
                <p class="note">
                    Tambahkan parameter username
                    untuk menampilkan preprocessing
                    preferensi pengguna.
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
                }}

                h2 {{
                    margin-top: 32px;
                    font-size: 20px;
                }}

                .note {{
                    padding: 12px;
                    background: #fff7ed;
                    border: 1px solid #fed7aa;
                    border-radius: 10px;
                }}

                .table-wrapper {{
                    overflow-x: auto;
                    padding: 16px;
                    margin-bottom: 28px;
                    background: white;
                    border-radius: 12px;
                }}

                table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                }}

                th {{
                    padding: 10px;
                    text-align: left;
                    white-space: nowrap;
                    background: #112D4E;
                    color: white;
                }}

                td {{
                    min-width: 180px;
                    max-width: 420px;
                    padding: 10px;
                    vertical-align: top;
                    border: 1px solid #e2e8f0;
                }}
            </style>
        </head>

        <body>
            <h1>Hasil Preprocessing</h1>

            <h2>Dokumen Buku</h2>
            <div class="table-wrapper">
                {book_table_html}
            </div>

            <h2>Dokumen Preferensi Pengguna</h2>
            <div class="table-wrapper">
                {user_table_html}
            </div>
        </body>
        </html>
        """

    except Exception as error:
        return jsonify({
            "success": False,
            "message": str(error)
        }), 500


# =========================================================
# REKOMENDASI
# =========================================================
@app.route(
    "/recommend",
    methods=["POST", "OPTIONS"]
)
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
            "message":
                "Username tidak ditemukan."
        }), 400

    books, tfidf, books_tfidf = (
        prepare_model()
    )

    pref = get_user_preferences(
        username
    )

    selected_titles = (
        parse_preference_list(
            pref.get(
                "buku_pilihan",
                pref.get(
                    "buku_favorit",
                    []
                )
            )
        )
    )

    preferred_subcategories = (
        parse_preference_list(
            pref.get(
                "sub_kategori",
                []
            )
        )
    )

    bookmarked_ids = (
        get_recent_bookmarks(
            username
        )
    )

    bookmarked_ids_for_profile = (
        bookmarked_ids[
            :MAX_BOOKMARKS_FOR_RECOMMENDATION
        ]
    )

    bookmarked_books = books[
        books["id"].isin(
            bookmarked_ids_for_profile
        )
    ]

    selected_books = (
        get_books_by_titles(
            books,
            selected_titles
        )
    )

    # Ground truth evaluasi.
    selected_subcategories = (
        selected_books[
            "subcategory"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .tolist()
    )

    relevance_subcategories = sorted(
        set(
            preferred_subcategories
            + selected_subcategories
        )
    )

    # Profil pengguna per field.
    user_field_texts = (
        build_user_field_texts(
            pref,
            selected_books,
            bookmarked_books
        )
    )

    similarity_scores = (
        weighted_cosine_similarity(
            user_field_texts,
            len(books),
            tfidf,
            books_tfidf,
            FIELD_WEIGHTS
        )
    )

    # Buku pilihan dan bookmark
    # tidak direkomendasikan kembali.
    all_bookmarked_books = books[
        books["id"].isin(
            bookmarked_ids
        )
    ]

    bookmarked_titles = (
        all_bookmarked_books[
            "title"
        ].tolist()
    )

    excluded_titles = {
        normalize_title(title)
        for title in (
            selected_titles
            + bookmarked_titles
        )
    }

    ranked_indices = (
        pd.Series(
            similarity_scores
        )
        .sort_values(
            ascending=False
        )
        .index
        .tolist()
    )

    results = []
    relevant_count = 0

    for index in ranked_indices:
        book = books.iloc[index]

        book_title = str(
            book["title"]
        ).strip()

        book_subcategory = str(
            book["subcategory"]
        ).strip()

        if (
            normalize_title(book_title)
            in excluded_titles
        ):
            continue

        is_relevant = (
            book_subcategory
            in relevance_subcategories
        )

        if is_relevant:
            relevant_count += 1

        results.append({
            "id":
                book["id"],
            "title":
                book_title,
            "author":
                book["author"],
            "cover":
                book["cover"],
            "subcategory":
                book_subcategory,
            "similarity":
                float(
                    similarity_scores[
                        index
                    ]
                ),
            "relevant":
                is_relevant
        })

        if len(results) >= 10:
            break

    total_recommended = len(results)

    precision = (
        relevant_count
        / total_recommended
        if total_recommended > 0
        else 0
    )

    candidate_books = books[
        ~books["title"]
        .apply(normalize_title)
        .isin(excluded_titles)
    ].copy()

    total_relevant = (
        candidate_books[
            candidate_books[
                "subcategory"
            ]
            .astype(str)
            .str.strip()
            .isin(
                relevance_subcategories
            )
        ]
        .shape[0]
    )

    recall = (
        relevant_count
        / total_relevant
        if total_relevant > 0
        else 0
    )

    return jsonify({
        "results": results,

        "evaluation": {
            "precision_at_10":
                precision,

            "recall_at_10":
                recall,

            "relevant_count":
                relevant_count,

            "total_recommended":
                total_recommended,

            "total_relevant_books":
                int(total_relevant),

            "relevance_subcategories":
                relevance_subcategories
        }
    })


# =========================================================
# RUN SERVER
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
