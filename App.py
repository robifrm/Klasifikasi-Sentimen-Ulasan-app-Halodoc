import streamlit as st # Library utama untuk membuat aplikasi web interaktif
import pandas as pd    # Untuk manipulasi data (membaca CSV, dll.)
import re              # Untuk operasi regular expression (pembersihan teks)
import nltk            # Natural Language Toolkit, untuk pemrosesan bahasa alami
from nltk.corpus import stopwords # Untuk daftar kata-kata umum yang tidak relevan (stopwords)
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory # Untuk melakukan stemming bahasa Indonesia
from sklearn.feature_extraction.text import CountVectorizer # Untuk mengubah teks menjadi representasi numerik
from sklearn.naive_bayes import MultinomialNB # Algoritma klasifikasi Naive Bayes
import joblib          # Untuk menyimpan dan memuat model Python
import os              # Untuk berinteraksi dengan sistem operasi (cek keberadaan file)

# Import tambahan yang diperlukan untuk split data dan SMOTE (penanganan imbalance data)
from sklearn.model_selection import train_test_split # Untuk membagi data menjadi training dan testing
from imblearn.over_sampling import SMOTE # Untuk menangani dataset yang tidak seimbang (opsional tapi disarankan)

# --- Pastikan NLTK Stopwords Tersedia ---
# Kode ini akan memeriksa apakah data stopwords NLTK sudah diunduh.
# Jika belum, ia akan mengunduhnya. Ini penting agar fungsi preprocessing berjalan.
try:
    nltk.data.find('corpora/stopwords')
# Catch the LookupError that indicates the resource is not found
except LookupError:
    st.info("Mengunduh data NLTK stopwords. Ini mungkin hanya terjadi sekali.")
    nltk.download('stopwords')
    st.success("Pengunduhan stopwords selesai!")
# Add a general exception catch for other potential download issues
except Exception as e:
    st.error(f"ERROR: Gagal mengunduh data NLTK stopwords: {e}")
    st.stop()


# --- Inisialisasi Preprocessing Teks ---
# Inisialisasi Stemmer Sastrawi
# Stemmer digunakan untuk mengubah kata menjadi bentuk dasarnya (misalnya "memakan" menjadi "makan").
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Inisialisasi Stopwords Bahasa Indonesia
# Stopwords adalah kata-kata umum yang biasanya tidak memiliki banyak makna untuk analisis sentimen
# (contoh: "yang", "dan", "di", dll.). Kata-kata ini akan dihapus.
list_stopwords = set(stopwords.words('indonesian'))

# --- Fungsi Preprocessing Teks ---
# Fungsi ini membersihkan teks ulasan dari noise sebelum dianalisis.
def clean_review_text(text):
    # 1. Case Folding: Mengubah semua teks menjadi huruf kecil untuk konsistensi.
    text = text.lower()
    # 2. Menghapus Angka: Menghapus semua digit angka dari teks.
    text = re.sub(r"\d+", "", text)
    # 3. Menghapus Tanda Baca: Menghapus semua karakter non-alfanumerik (kecuali spasi).
    text = re.sub(r'[^\w\s]', '', text)
    # 4. Menghapus Spasi Berlebih: Mengganti beberapa spasi menjadi satu spasi tunggal dan menghapus spasi di awal/akhir.
    text = re.sub(r'\s+', ' ', text).strip()
    # 5. Stemming: Mengubah kata menjadi bentuk dasarnya.
    text = stemmer.stem(text)
    # 6. Menghapus Stopwords: Menghapus kata-kata yang ada di daftar stopwords.
    words = text.split() # Memecah teks menjadi kata-kata
    words_clean = [word for word in words if word not in list_stopwords] # Filter stopwords
    text = " ".join(words_clean) # Menggabungkan kembali kata-kata menjadi string
    return text

# --- Path untuk Menyimpan/Memuat Model ---
# Ini adalah lokasi file tempat model Naive Bayes dan Vectorizer akan disimpan atau dimuat.
model_path = 'naive_bayes_model.pkl'
vectorizer_path = 'count_vectorizer.pkl'

# --- Logika Pelatihan atau Pemuatan Model ---
# Bagian ini sangat penting:
# Jika file model (`.pkl`) belum ada, aplikasi akan mencoba melatih model dari awal.
# Ini berguna jika Anda menjalankan aplikasi untuk pertama kali dan belum memiliki model yang tersimpan.
# Jika file model sudah ada, aplikasi akan langsung memuat model yang sudah dilatih,
# sehingga tidak perlu melatih ulang setiap kali aplikasi dijalankan.
if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    st.warning("File model atau vectorizer tidak ditemukan. Melatih model sekarang. Proses ini mungkin memakan waktu!")
    try:
        # Load Data: Membaca dataset ulasan dari file CSV.
        df = pd.read_csv("Review Aplikasi Halodoc.csv")

        # Membuat Kolom Sentimen: Mengubah rating menjadi label sentimen (1=Positif, 0=Negatif).
        # Diasumsikan rating >= 3 adalah positif, selain itu negatif.
        df['sentimen'] = df['rating'].apply(lambda x: 1 if x >= 3 else 0)

        # Preprocessing Teks Data Latih: Menerapkan fungsi clean_review_text ke setiap ulasan.
        df['review_bersih'] = df['review'].apply(clean_review_text)

        # Split Data: Memisahkan data menjadi fitur (X) dan label (y).
        # Kemudian membagi data menjadi set pelatihan (training) dan pengujian (testing).
        X = df['review_bersih']
        y = df['sentimen']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Vectorization: Mengubah teks bersih menjadi representasi numerik (CountVector).
        # `tokenizer=lambda x: x.split()`: Menggunakan spasi sebagai pemisah kata.
        # `min_df=5`: Hanya kata-kata yang muncul minimal 5 kali akan dipertimbangkan.
        vectorizer = CountVectorizer(tokenizer=lambda x: x.split(), min_df=5)
        X_train_vec = vectorizer.fit_transform(X_train) # Fit dan transform data training
        X_test_vec = vectorizer.transform(X_test)     # Hanya transform data testing

        # SMOTE (Synthetic Minority Over-sampling Technique):
        # Digunakan untuk menyeimbangkan jumlah sampel kelas minoritas dalam dataset,
        # jika ada ketidakseimbangan antara jumlah ulasan positif dan negatif.
        smote = SMOTE(random_state=42)
        X_train_smote, y_train_smote = smote.fit_resample(X_train_vec, y_train)

        # Training Model Naive Bayes: Melatih model klasifikasi.
        naive_bayes = MultinomialNB()
        naive_bayes.fit(X_train_smote, y_train_smote)

        # Simpan Model dan Vectorizer: Menyimpan model dan vectorizer agar bisa digunakan kembali.
        joblib.dump(naive_bayes, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        st.success("Model dan vectorizer berhasil dilatih dan disimpan!")

    except FileNotFoundError:
        st.error("ERROR: File 'Review Aplikasi Halodoc.csv' tidak ditemukan. Pastikan file CSV ini berada di direktori yang sama dengan aplikasi Streamlit Anda.")
        st.stop() # Menghentikan aplikasi jika file tidak ditemukan
    except Exception as e:
        st.error(f"ERROR: Terjadi kesalahan saat melatih model: {e}")
        st.stop()
else:
    st.info("Memuat model dan vectorizer yang sudah ada dari file.")

# --- Memuat Model dan Vectorizer (setelah pelatihan atau jika sudah ada) ---
# Bagian ini akan selalu mencoba memuat model yang sudah disimpan.
try:
    naive_bayes = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
except Exception as e:
    st.error(f"ERROR: Gagal memuat model atau vectorizer. Pastikan Anda telah melatih dan menyimpannya dengan benar atau file ada di lokasi yang tepat. Error: {e}")
    st.stop() # Menghentikan aplikasi jika gagal memuat model

# --- Aplikasi Streamlit Utama ---
st.title("Aplikasi Analisis Sentimen Ulasan Halodoc")
st.markdown("Aplikasi ini menganalisis sentimen (positif/negatif) dari ulasan aplikasi Halodoc.")

# Input Teks dari Pengguna: Kotak teks tempat pengguna bisa memasukkan ulasan.
user_input = st.text_area("Masukkan ulasan yang ingin Anda analisis:", "")

# Tombol untuk Menganalisis Sentimen
if st.button("Analisis Sentimen"):
    if user_input:
        # Preprocessing Teks Input Pengguna: Membersihkan teks yang dimasukkan pengguna.
        cleaned_input = clean_review_text(user_input)

        # Vectorize Teks Input Pengguna: Mengubah teks bersih menjadi representasi numerik.
        # Penting: Gunakan `transform` (bukan `fit_transform`) karena vectorizer sudah dilatih.
        input_vec = vectorizer.transform([cleaned_input])

        # Prediksi Sentimen: Model memprediksi apakah ulasan positif (1) atau negatif (0).
        prediction = naive_bayes.predict(input_vec)

        # Tampilkan Hasil: Menampilkan sentimen yang diprediksi.
        st.subheader("Hasil Analisis:")
        if prediction[0] == 1:
            st.success("Sentimen: POSITIF 👍") # Menampilkan pesan sukses untuk sentimen positif
        else:
            st.error("Sentimen: NEGATIF 👎") # Menampilkan pesan error untuk sentimen negatif

        st.write(f"**Teks Asli:** {user_input}")
        st.write(f"**Teks Bersih (setelah preprocessing):** {cleaned_input}")
    else:
        st.warning("Mohon masukkan teks ulasan untuk dianalisis terlebih dahulu.")

st.markdown("---")
st.markdown(
    """
    *Aplikasi ini dikembangkan untuk demonstrasi analisis sentimen ulasan.*
    *Menggunakan Model Klasifikasi Naive Bayes dengan Framework Streamlit.*
    *Pengembang: M. Robi Firmansyah*
    """
)
