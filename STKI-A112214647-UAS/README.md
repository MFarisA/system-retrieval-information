
# Proyek Analisis Sentimen

Proyek ini dirancang untuk melakukan analisis sentimen terhadap ulasan pengguna untuk aplikasi **Genshin Impact**. Proyek ini terdiri dari tiga notebook Jupyter utama, yang masing-masing bertanggung jawab untuk bagian tertentu dalam alur kerja: pengambilan data, pra-pemrosesan, dan prediksi.

## Struktur Proyek

1. **serpAPI.ipynb**: Mengambil ulasan aplikasi dari Google Play Store menggunakan `google_play_scraper`.
2. **pre-process.ipynb**: Mempersiapkan data yang diambil dengan membersihkan dan memproses teks, membuat word clouds, dan melatih model pembelajaran mesin untuk klasifikasi sentimen.
3. **predict.ipynb**: Menggunakan model yang dilatih untuk memprediksi sentimen dari input teks baru.

---

## 1. `serpAPI.ipynb` - Mengambil Ulasan

Notebook ini mengambil ulasan dari aplikasi **Genshin Impact** di Google Play Store menggunakan pustaka `google_play_scraper`.

### Penjelasan Kode:
- **Impor Pustaka**: Notebook ini mengimpor modul `csv` untuk menyimpan data dan pustaka `google_play_scraper` untuk mengambil ulasan aplikasi.
- **Mengambil Ulasan**: Fungsi `reviews()` digunakan untuk mengumpulkan 1000 ulasan aplikasi Genshin Impact dari Google Play Store, dengan parameter yang menentukan ID aplikasi (`com.miHoYo.GenshinImpact`), bahasa (`en`), dan negara (`us`).
- **Menyimpan Data**: Ulasan disimpan dalam file CSV (`genshin_impact_reviews.csv`). Kolom dalam CSV ini mencakup:
  - `id`: ID Ulasan
  - `title`: Nama pengguna yang mengulas
  - `avatar`: Gambar avatar (tidak tersedia dalam dataset ini)
  - `rating`: Rating (dari 1 hingga 5 bintang)
  - `snippet`: Isi teks dari ulasan
  - `likes`: Jumlah suka yang diterima ulasan
  - `date`: Tanggal dan waktu saat ulasan diposting
  - `iso_date`: Tanggal dalam format ISO 8601
  - `response`: Tanggapan pengembang (jika ada)

### Output:
Ulasan disimpan dalam file CSV bernama `genshin_impact_reviews.csv`.

---

## 2. `pre-process.ipynb` - Pra-pemrosesan dan Pelatihan Model

Notebook ini memproses ulasan yang diambil, membersihkan teks, dan melatih model analisis sentimen.

### Penjelasan Kode:
- **Memuat Data**: Memuat data ulasan dari file `genshin_impact_reviews.csv`.
- **Pra-pemrosesan Teks**:
  - Menghapus karakter yang tidak perlu (misalnya simbol khusus, URL).
  - Men-tokenisasi teks ulasan dan menghapus kata-kata umum (stopwords).
  - Melakukan lemmatization untuk mengubah kata menjadi bentuk dasar.
- **Visualisasi**:
  - **WordCloud**: Membuat word clouds untuk memvisualisasikan kata-kata yang paling sering muncul pada komentar positif dan negatif.
  - **Bar Plot**: Membuat bar plot untuk menunjukkan 10 kata paling sering dalam komentar positif dan negatif.
- **Pelatihan Model**:
  - **TfidfVectorizer**: Mengubah teks yang sudah dibersihkan menjadi fitur numerik (nilai TF-IDF).
  - **Model XGBoost**: Melatih model klasifikasi sentimen menggunakan **XGBoost** (algoritma gradient boosting).
- **Menyimpan Model**:
  - Menyimpan model yang telah dilatih (`xgb_model.pkl`).
  - Menyimpan vectorizer (`vectorizer.pkl`) yang digunakan untuk mengubah data input baru.

### Output:
Model yang dilatih dan vectorizer disimpan dalam file `.pkl`.

---

## 3. `predict.ipynb` - Prediksi

Notebook ini menggunakan model yang dilatih dan vectorizer untuk memprediksi sentimen dari input teks baru.

### Penjelasan Kode:
- **Memuat Model dan Vectorizer**: Memuat model XGBoost yang dilatih (`xgb_model.pkl`) dan vectorizer (`vectorizer.pkl`) dari file `.pkl`.
- **Prediksi**:
  - Menerima input teks baru (misalnya, "good graphic").
  - Mengubah input menggunakan vectorizer.
  - Membuat prediksi menggunakan model yang dilatih (sentimen positif atau negatif).
- **Output**:
  - Menampilkan sentimen (positif atau negatif) dari teks yang diberikan.

### Output:
Hasil prediksi ditampilkan dalam sel output yang menunjukkan sentimen untuk teks yang diberikan.

---

## Dataset

Dataset ini terdiri dari ulasan pengguna untuk aplikasi **Genshin Impact** yang diambil dari Google Play Store. Dataset disimpan dalam format CSV dengan kolom-kolom berikut:

- **id**: ID unik untuk setiap ulasan.
- **title**: Nama pengguna yang memberikan ulasan.
- **avatar**: Gambar avatar pengguna (tidak tersedia dalam dataset).
- **rating**: Rating yang diberikan oleh pengguna (dari 1 hingga 5 bintang).
- **snippet**: Isi teks dari ulasan.
- **likes**: Jumlah suka yang diterima ulasan.
- **date**: Tanggal dan waktu ulasan diposting.
- **iso_date**: Tanggal dalam format ISO 8601.
- **response**: Tanggapan pengembang terhadap ulasan (jika ada).

Contoh dari dataset:

| id                                | title                  | rating | snippet                                                                 | likes | date                  | response                      |
|-----------------------------------|------------------------|--------|-------------------------------------------------------------------------|-------|-----------------------|-------------------------------|
| 45b29182-3368-445f-8353-efc25383ea9f | Francesca Ashley Inguito | 1      | This was awful, the dialogue of characters in the game was terrible.   | 1     | 2025-01-12 17:37:21   | NaN                           |
| 5baab1b9-0540-4ac3-90c9-1ee3462bdc73 | Ikmal Hariz            | 5      | MANYAK BAGUS LA I LIKE CUMA PLS CEPAT KAN SIKIT                       | 0     | 2025-01-12 02:42:05   | NaN                           |

---

## Visualisasi

Di bawah ini adalah visualisasi dari analisis sentimen pada komentar positif dan negatif menggunakan word clouds dan bar plots.

![Visualisasi Analisis Sentimen](sandbox:image.jpeg)

---

## Prasyarat

Untuk menjalankan notebook-notebook ini, Anda perlu menginstal pustaka Python berikut:
- `pandas`
- `numpy`
- `scikit-learn`
- `imbalanced-learn`
- `spacy`
- `nltk`
- `xgboost`
- `emoji`
- `matplotlib`
- `wordcloud`
- `google-play-scraper`

Anda dapat menginstalnya menggunakan perintah berikut:
```bash
pip install pandas numpy scikit-learn imbalanced-learn spacy nltk xgboost emoji matplotlib wordcloud google-play-scraper
```

---

## Menjalankan Proyek

1. Mulailah dengan menjalankan `serpAPI.ipynb` untuk mengambil ulasan dan menyimpannya sebagai CSV.
2. Kemudian jalankan `pre-process.ipynb` untuk memproses ulasan dan melatih model analisis sentimen.
3. Terakhir, jalankan `predict.ipynb` untuk membuat prediksi sentimen pada teks baru.

---

## Kesimpulan

Proyek ini mendemonstrasikan alur lengkap untuk melakukan analisis sentimen pada ulasan aplikasi, mulai dari pengambilan data dan pra-pemrosesan hingga pelatihan model dan prediksi. Model yang dilatih dapat digunakan kembali untuk menganalisis teks yang diberikan dan memprediksi apakah sentimennya positif atau negatif.

