API_BASE_URL = "https://pojokbaca-brida.my.id/api"

from flask import Flask, request, jsonifyfrom flask_cors import CORS

import reimport nltkimport requestsimport pandas as pd

from functools import lru_cachefrom Sastrawi.Stemmer.StemmerFactory import StemmerFactoryfrom nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizerfrom sklearn.metrics.pairwise import cosine_similarity

=========================

INISIALISASI APLIKASI

=========================

app = Flask(name)

CORS(app, origins=["https://pojokbaca-brida.my.id","https://www.pojokbaca-brida.my.id"])

nltk.download("stopwords", quiet=True)

stop_words_idn = set(stopwords.words("indonesian"))

factory = StemmerFactory()stemmer = factory.create_stemmer()



=========================

STEMMING CACHE

=========================

@lru_cache(maxsize=50000)def stem_cached(word):return stemmer.stem(word)



=========================

ROUTE UTAMA

=========================

@app.route("/", methods=["GET"])def home():return jsonify({"success": True,"message": "Flask recommendation API is running"})



=========================

PREPROCESSING

=========================

def preprocess(text):text = str(text).lower()

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



=========================

TABEL PREPROCESSING

=========================

def clean_text_for_table(text):text = str(text)text = re.sub(r"\r\n|\n|\r", " ", text)text = re.sub(r"['’‘`´]", "", text)text = re.sub(r"[^\w\s]", " ", text)text = re.sub(r"\s+", " ", text).strip()return text

def get_preprocessing_steps(text):# 1. Case foldingcase_folding = str(text).lower()

# 2. Punctuation removal
punctuation_removal = clean_text_for_table(case_folding)

# 3. Tokenizing
tokenizing = punctuation_removal.split()

# 4. Stopword removal
stopword_removal = [
    word for word in tokenizing
    if word not in stop_words_idn and len(word) > 2
]

# 5. Stemming
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

book table

def generate_preprocessing_tables():url = f"{API_BASE_URL}/books.php"response = requests.get(url)books = pd.DataFrame(response.json())

books["id"] = books["id"].astype(str)

# =========================
# AMBIL BUKU ID 34 DULU
# =========================
target_book = books[books["id"] == "34"].copy()

other_books = books[books["id"] != "34"].head(5).copy()

# gabungkan, id 34 ada di paling atas
df = pd.concat([target_book, other_books], ignore_index=True)

# =========================
# DOKUMEN HASIL MERGE DATA
# =========================
df["merged_text"] = (
    df["title"].astype(str) + " " +
    df["subcategory"].astype(str) + " " +
    df["author"].astype(str) + " " +
    df["category"].astype(str) + " " +
    df["sinopsis"].astype(str)
)

# =========================
# DOKUMEN HASIL PEMBOBOTAN ATRIBUT
# disamakan dengan load_books()
# =========================
df["weighted_text"] = (
    (df["subcategory"].astype(str) + " ") * 3 +
    (df["title"].astype(str) + " ") * 2 +
    (df["author"].astype(str) + " ") * 1 +
    (df["category"].astype(str) + " ") * 1 +
    (df["sinopsis"].astype(str) + " ") * 1
)

# =========================
# PREPROCESSING weighted_text
# =========================
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

user table

def generate_user_preprocessing_table(username, books):if not username:return pd.DataFrame()

books = books.copy()
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

# ambil bookmark user
bookmarked_book_ids = get_recent_bookmarks(username)

bookmarked_books = books[
    books["id"].isin(bookmarked_book_ids)
]

# ambil buku favorit awal dari form
survey_favorite_books_df = get_books_by_titles(
    books,
    survey_favorite_books
)

bookmarked_text = build_books_text(
    bookmarked_books,
    include_sinopsis=False
)

survey_favorite_text = build_books_text(
    survey_favorite_books_df,
    include_sinopsis=False
)

# dokumen user sebelum bobot
merged_user_text = (
    " ".join(preferred_subcategories) + " " +
    survey_favorite_text + " " +
    bookmarked_text + " " +
    " ".join(preferred_categories)
)

# dokumen user setelah bobot
# ini disamakan dengan route /recommend kamu
weighted_user_text = (
    (" ".join(preferred_subcategories) + " ") * 3 +
    (survey_favorite_text + " ") * 2 +
    (bookmarked_text + " ") * 1 +
    (" ".join(preferred_categories) + " ") * 1
)

steps = get_preprocessing_steps(weighted_user_text)

survey_favorite_subcategories = (
    survey_favorite_books_df["subcategory"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
    if not survey_favorite_books_df.empty else []
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

    "Buku Pilihan Awal": ", ".join(survey_favorite_books),
    "Subkategori Buku Pilihan Awal": ", ".join(survey_favorite_subcategories),

    "Buku Favorit Katalog": ", ".join(bookmarked_titles),
    "Subkategori Buku Favorit Katalog": ", ".join(bookmarked_subcategories),

    "Dokumen Hasil Merge Data": merged_user_text,
    "Dokumen Hasil Pembobotan Atribut": weighted_user_text,

    **steps
}

return pd.DataFrame([user_row])



@app.route("/preprocessing-table", methods=["GET"])def preprocessing_table():try:username = request.args.get("username", "").strip()

    book_df, books = generate_preprocessing_tables()

    user_df = generate_user_preprocessing_table(
        username,
        books
    )

    # biar list token enak dibaca
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

=========================

PEMBENTUKAN TEKS BUKU

=========================

def build_book_text(book, include_sinopsis=True):text = (str(book.get("title", "")) + " " +str(book.get("subcategory", "")) + " " +str(book.get("author", "")) + " " +str(book.get("category", "")))

if include_sinopsis:
    text += " " + str(book.get("sinopsis", ""))

return text





def clean_list(values):if values is None:return []

if isinstance(values, list):
    raw_values = values
else:
    raw_values = [values]

return [
    str(value).strip()
    for value in raw_values
    if str(value).strip()
]



=========================

LOAD DATA BUKU

=========================

def load_books():url = f"{API_BASE_URL}/books.php"response = requests.get(url)books = pd.DataFrame(response.json())

books["combined"] = (
    (books["subcategory"].astype(str) + " ") * 3 +
    (books["title"].astype(str) + " ") * 2 +
    (books["author"].astype(str) + " ") * 1 +
    (books["category"].astype(str) + " ") * 1 + 
    (books["sinopsis"].astype(str) + " ") * 1
)

books["hasil"] = books["combined"].apply(preprocess)

return books



=========================

MODEL CACHE

=========================

books_cache = Nonetfidf_cache = Nonebooks_tfidf_cache = None

def prepare_model(force_refresh=False):global books_cache, tfidf_cache, books_tfidf_cache

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
    min_df=1,
    sublinear_tf=True
)

books_tfidf = tfidf.fit_transform(books["hasil"])

books_cache = books
tfidf_cache = tfidf
books_tfidf_cache = books_tfidf

return books_cache, tfidf_cache, books_tfidf_cache

@app.route("/refresh-model", methods=["GET"])def refresh_model():prepare_model(force_refresh=True)

return jsonify({
    "success": True,
    "message": "Model rekomendasi berhasil diperbarui."
})



=========================

PREFERENSI USER

=========================

def get_user_preferences(username):response = requests.get(f"{API_BASE_URL}/get_preferences.php?username={username}")

return response.json()

data bookmark

def get_recent_bookmarks(username):response = requests.post(f"{API_BASE_URL}/get_recent_favorites.php",json={"username": username})

bookmarked_book_ids = response.json() or []

return [
    str(book_id)
    for book_id in bookmarked_book_ids
]
# (judul, subk, penulis, kateg, sinopsis)

normalisasi buku favorit/bookmark

def normalize_title(text):text = str(text).lower().strip()text = re.sub(r"\s+", " ", text)text = re.sub(r"\s*:\s*", ":", text)return text

def get_books_by_titles(books, titles):normalized_titles = [normalize_title(title)for title in titles]

return books[
    books["title"].apply(normalize_title).isin(normalized_titles)
]

def build_books_text(books_df, include_sinopsis=True):text = ""

for _, book in books_df.iterrows():
    text += " " + build_book_text(
        book,
        include_sinopsis=include_sinopsis
    )

return text



=========================

REKOMENDASI

=========================

@app.route("/recommend", methods=["POST", "OPTIONS"])def recommend():if request.method == "OPTIONS":return jsonify({"success": True}), 200

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
    (" ".join(preferred_subcategories) + " ") * 3 +
    
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



=========================

RUN SERVER

=========================

if name == "main":app.run(debug=True)
