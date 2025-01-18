import streamlit as st
import joblib
import os

# Menentukan path file model dan vectorizer berdasarkan direktori file
model_path = os.path.join(os.path.dirname(__file__), 'xgb_model.pkl')
vectorizer_path = os.path.join(os.path.dirname(__file__), 'vectorizer.pkl')

# Memuat model dan vectorizer dari file .pkl
loaded_model = joblib.load(model_path)
loaded_vectorizer = joblib.load(vectorizer_path)

# Fungsi utama aplikasi
def main():
    # Judul Aplikasi
    st.title("Sentiment Analysis Using Logistic Regression")

    # Deskripsi aplikasi
    st.write("""
        Aplikasi ini melakukan analisis sentimen pada teks menggunakan model Logistic Regression yang telah dilatih.
        Masukkan teks untuk menganalisis apakah teks tersebut bersentimen **Positif** atau **Negatif**.
    """)

    # Input teks dari pengguna
    new_text = st.text_input("Masukkan teks untuk analisis sentimen:")

    # Jika tombol analisis ditekan
    if st.button("Analisis Sentimen"):
        if new_text:
            # Mengubah teks menjadi fitur menggunakan vectorizer
            new_text_tfidf = loaded_vectorizer.transform([new_text])

            # Melakukan prediksi dengan model
            prediction = loaded_model.predict(new_text_tfidf)

            # Menampilkan hasil prediksi
            if prediction == 1:
                st.write(f"**Prediksi untuk teks '{new_text}': Positif**")
            else:
                st.write(f"**Prediksi untuk teks '{new_text}': Negatif**")
        else:
            st.write("Mohon masukkan teks untuk analisis.")

# Mengecek jika file ini dijalankan langsung
if __name__ == "__main__":
    main()