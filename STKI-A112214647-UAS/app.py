import streamlit as st
import pickle

# Memuat model dan vectorizer dari file .pkl
with open('xgb_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Fungsi utama aplikasi
def main():
    # Judul Aplikasi
    st.title("Sentiment Analysis Using XGBoost")

    # Deskripsi aplikasi
    st.write("""
        Aplikasi ini melakukan analisis sentimen pada teks menggunakan model XGBoost yang telah dilatih.
        Masukkan teks untuk menganalisis apakah teks tersebut bersentimen **Positif** atau **Negatif**.
    """)

    # Input teks dari pengguna
    new_text = st.text_input("Masukkan teks untuk analisis sentimen:")

    # Jika tombol analisis ditekan
    if st.button("Analisis Sentimen"):
        if new_text:
            # Mengubah teks menjadi fitur menggunakan vectorizer
            new_text_tfidf = vectorizer.transform([new_text])

            # Melakukan prediksi dengan model
            prediction = model.predict(new_text_tfidf)

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
