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
# KONFIGURASI FIELD & BOBOT
# =========================
# PERUBAHAN: bobot sekarang didefinisikan di satu tempat dan dipakai
# saat MENGGABUNGKAN skor cosine similarity per-field, bukan saat
# membangun teks dokumen (tidak ada lagi pengulangan string * 2 / * 1).
FIELD_COLUMNS = ["subcategory", "title", "author", "category", "sinopsis"]

FIELD_WEIGHTS = {
    "subcategory": 2,
    "title": 2,
    "author": 1,
    "category": 1,
    "sinopsis": 1,
}


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


# PERUBAHAN: build_book_text() dan build_books_text() DIHAPUS.
# Fungsi ini dulu dipakai untuk menggabungkan beberapa field buku
# (judul + subkategori + penulis + kategori) menjadi SATU teks per buku.
# Karena sekarang tiap field dihitung similarity-nya secara terpisah,
# menggabungkan field seperti ini sudah tidak relevan lagi.


# =========================
# LOAD DATA BUKU
# =========================
def load_books():
    url = f"{API_BASE_URL}/books.php"
    response = requests.get(url)
    books = pd.DataFrame(response.json())

    # PERUBAHAN: tidak ada lagi pembobotan lewat pengulangan string.
    # Tiap atribut buku di-preprocess SENDIRI-SENDIRI (tidak digabung),
    # karena tiap field butuh TF-IDF vector space-nya masing-masing
    # untuk bisa dihitung cosine similarity per field.
    for field in FIELD_COLUMNS:
        books[f"{field}_clean"] = (
            books[field].astype(str).apply(preprocess)
        )

    return books


# =========================
# MODEL TF-IDF (PER FIELD)
# =========================
# PERUBAHAN: dulu hanya ADA SATU TfidfVectorizer + SATU matrix TF-IDF
# (books_tfidf) karena semua field digabung jadi satu dokumen.
# Sekarang kita butuh SATU TfidfVectorizer + SATU matrix TF-IDF UNTUK
# TIAP FIELD, karena bobot diterapkan di tahap similarity per field.
books_cache = None
tfidf_cache = None        # dict: {field: TfidfVectorizer}
books_tfidf_cache = None  # dict: {field: sparse matrix TF-IDF}


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

    tfidf_cache = {}
    books_tfidf_cache = {}

    for field in FIELD_COLUMNS:
        # PERUBAHAN: min_df diturunkan dari 2 menjadi 1.
        # Alasan: dulu min_df=2 dihitung di atas SATU korpus besar
        # (gabungan 5 field), jadi cukup aman membuang term yang
        # hanya muncul di 1 dokumen. Sekarang tiap field punya
        # korpus sendiri yang jauh lebih kecil/sempit (contoh: field
        # "author" atau "category" seringkali kata-katanya unik per
        # baris), sehingga min_df=2 berisiko membuat vocabulary
        # kosong atau membuang term pembeda yang justru penting.
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(books[f"{field}_clean"])

        tfidf_cache[field] = vectorizer
        books_tfidf_cache[field] = matrix

    books_cache = books

    return books_cache, tfidf_cache, books_tfidf_cache


@app.route("/refresh-model", methods=["GET"])
def refresh_model():
    prepare_model(force_refresh=True)

    return jsonify({
        "success": True,
        "message": "Model rekomendasi berhasil diperbarui."
    })


# =========================
# WEIGHTED COSINE SIMILARITY
# =========================
# PERUBAHAN / TAMBAHAN: fungsi baru. Di sinilah bobot field benar-benar
# diterapkan sekarang -- bukan lagi lewat pengulangan string, tapi lewat
# rata-rata berbobot dari cosine similarity tiap field:
#
#   skor_akhir = (w1*sim_field1 + w2*sim_field2 + ...) / (w1 + w2 + ...)
#
# Field yang tidak punya sinyal dari sisi pengguna (teks kosong)
# dilewati, supaya tidak mengurangi skor secara tidak adil.
def weighted_cosine_similarity(user_field_texts, n_books, tfidf_dict, books_tfidf_dict, weights):
    total_score = None
    total_weight_used = 0

    for field, weight in weights.items():
        raw_text = user_field_texts.get(field, "")

        if not raw_text.strip():
            # Tidak ada sinyal preferensi pengguna untuk field ini
            # (contoh: field "sinopsis", karena pengguna tidak pernah
            # mengisi preferensi berbentuk sinopsis).
            continue

        vectorizer = tfidf_dict[field]
        books_matrix = books_tfidf_dict[field]

        user_vec = vectorizer.transform([preprocess(raw_text)])
        field_sim = cosine_similarity(user_vec, books_matrix).flatten()

        if total_score is None:
            total_score = weight * field_sim
        else:
            total_score += weight * field_sim

        total_weight_used += weight

    if total_score is None or total_weight_used == 0:
        return [0.0] * n_books

    return total_score / total_weight_used


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


def build_user_field_texts(pref, selected_books_df, bookmarked_books):
    """
    PERUBAHAN / TAMBAHAN: pengganti user_text tunggal yang dulu dibangun
    dengan pengulangan string. Sekarang menghasilkan teks preferensi
    TERPISAH per field, supaya bisa dibandingkan ke TF-IDF field yang
    sesuai di weighted_cosine_similarity().
    """
    preferred_subcategories = parse_preference_list(pref.get("sub_kategori", []))
    preferred_categories = parse_preference_list(pref.get("kategori", []))

    return {
        "subcategory": " ".join(
            preferred_subcategories
            + selected_books_df["subcategory"].astype(str).tolist()
            + bookmarked_books["subcategory"].astype(str).tolist()
        ),
        "title": " ".join(
            selected_books_df["title"].astype(str).tolist()
            + bookmarked_books["title"].astype(str).tolist()
        ),
        "author": " ".join(
            selected_books_df["author"].astype(str).tolist()
            + bookmarked_books["author"].astype(str).tolist()
        ),
        "category": " ".join(
            preferred_categories
            + selected_books_df["category"].astype(str).tolist()
            + bookmarked_books["category"].astype(str).tolist()
        ),
        # PERUBAHAN: field "sinopsis" sengaja dikosongkan karena tidak
        # pernah ada sinyal sinopsis dari sisi pengguna (baik dulu maupun
        # sekarang). Bedanya: dulu sinopsis buku tetap ikut campur di
        # SATU vektor gabungan, sehingga bisa sedikit "mengencerkan"
        # skor cosine (menambah panjang vektor buku tanpa ada yang
        # dicocokkan). Sekarang field ini otomatis dilewati di
        # weighted_cosine_similarity(), jadi efek pengenceran itu hilang
        # -- ini salah satu alasan skor akhir bisa sedikit berbeda dari
        # versi sebelumnya.
        "sinopsis": "",
    }


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

    # PERUBAHAN: dulu ada dua versi teks per buku ("merged_text" tanpa
    # bobot dan "weighted_text" dengan pengulangan string). Sekarang
    # tidak ada lagi versi "weighted", karena bobot tidak lagi
    # diterapkan di tahap ini. Tabel demonstrasi menampilkan preprocessing
    # PER FIELD, sesuai alur baru.
    preprocessing_rows = []

    for _, row in df.iterrows():
        for field in FIELD_COLUMNS:
            steps = get_preprocessing_steps(row.get(field, ""))

            preprocessing_rows.append({
                "Jenis Dokumen": "Buku",
                "ID Buku": row.get("id", ""),
                "Judul Buku": row.get("title", ""),
                "Field": field,
                "Teks Asli Field": row.get(field, ""),
                **steps
            })

    return pd.DataFrame(preprocessing_rows), books


def generate_user_preprocessing_table(username, books):
    if not username:
        return pd.DataFrame()

    books = books.copy()
    books["id"] = books["id"].astype(str)

    pref = get_user_preferences(username)

    preference_selected_books = parse_preference_list(
        pref.get("buku_pilihan", pref.get("buku_favorit", []))
    )

    bookmarked_book_ids = get_recent_bookmarks(username)
    bookmarked_book_ids_for_recommendation = bookmarked_book_ids[
        :MAX_BOOKMARKS_FOR_RECOMMENDATION
    ]
    bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids_for_recommendation)
    ]

    selected_books_df = get_books_by_titles(books, preference_selected_books)

    # PERUBAHAN: memakai fungsi baru build_user_field_texts() supaya
    # tabel contoh preprocessing konsisten dengan apa yang benar-benar
    # dipakai saat menghitung rekomendasi di /recommend.
    user_field_texts = build_user_field_texts(
        pref,
        selected_books_df,
        bookmarked_books
    )

    preprocessing_rows = []

    for field in FIELD_COLUMNS:
        text = user_field_texts.get(field, "")
        steps = get_preprocessing_steps(text)

        preprocessing_rows.append({
            "Jenis Dokumen": "Preferensi Pengguna",
            "Username": username,
            "Field": field,
            "Teks Asli Field": text,
            "Bobot Field": FIELD_WEIGHTS.get(field),
            **steps
        })

    return pd.DataFrame(preprocessing_rows)


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
                Data berikut menampilkan tahapan preprocessing PER FIELD (judul, subkategori,
                penulis, kategori, sinopsis). Bobot field tidak lagi diterapkan di tahap ini --
                bobot baru dipakai nanti saat menggabungkan skor cosine similarity tiap field.
            </p>

            <h2>1. Preprocessing Dokumen Buku (per field)</h2>
            <div class="table-wrapper">
                {book_table_html}
            </div>

            <h2>2. Preprocessing Dokumen Preferensi Pengguna (per field)</h2>
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

    # PERUBAHAN: tfidf dan books_tfidf sekarang berupa dict per field,
    # bukan lagi satu vectorizer + satu matrix.
    books, tfidf, books_tfidf = prepare_model()

    books["id"] = books["id"].astype(str)

    pref = get_user_preferences(username)

    preference_selected_books = parse_preference_list(
        pref.get("buku_pilihan", pref.get("buku_favorit", []))
    )

    preferred_subcategories = parse_preference_list(pref.get("sub_kategori", []))
    preferred_categories = parse_preference_list(pref.get("kategori", []))

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

    # Acuan relevansi evaluasi -- TIDAK BERUBAH dari versi sebelumnya
    selected_book_subcategories = parse_preference_list(
        selected_books_df["subcategory"].dropna().tolist()
    )

    relevance_subcategories = sorted(set(
        preferred_subcategories +
        selected_book_subcategories
    ))

    # PERUBAHAN: profil pengguna sekarang berupa dict per field,
    # bukan satu string weighted_user_text.
    user_field_texts = build_user_field_texts(
        pref,
        selected_books_df,
        bookmarked_books
    )

    # PERUBAHAN INTI: cosine similarity dihitung per field lalu
    # digabung berbobot -- bukan satu kali cosine_similarity(user_vec, books_tfidf).
    sim_scores = weighted_cosine_similarity(
        user_field_texts,
        books.shape[0],
        tfidf,
        books_tfidf,
        FIELD_WEIGHTS
    )

    # Buku pilihan dan semua bookmark tidak ditampilkan ulang -- TIDAK BERUBAH
    all_bookmarked_books = books[
        books["id"].isin(bookmarked_book_ids)
    ]

    bookmarked_book_titles = all_bookmarked_books["title"].tolist()

    excluded_book_titles = [
        normalize_title(title)
        for title in (preference_selected_books + bookmarked_book_titles)
    ]

    top_idx = list(pd.Series(sim_scores).sort_values(ascending=False).index)

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
