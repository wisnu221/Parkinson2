import numpy as np
import pandas as pd
import streamlit as st
import pickle

# --- Load the trained model ---
# Pastikan 'parkinsons_model.sav' berada di direktori yang sama dengan 'app.py'
try:
    loaded_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except FileNotFoundError:
    st.error("Error: 'parkinsons_model.sav' not found. Please ensure the model file is in the same directory.")
    st.stop() # Stop the app if the model isn't found

# --- List of features with their corresponding descriptions and default values ---
# Menggunakan dictionary untuk menyimpan nama fitur, deskripsi, dan nilai default
# Nilai default ini bisa disesuaikan dengan mean atau median dari dataset Anda
# atau nilai yang masuk akal secara medis.
features_info = {
    "MDVP:Fo(Hz)": {"label": "1. MDVP:Fo(Hz)", "description": "Rata-rata frekuensi fundamental vokal.", "default": 154.23, "format": "%.3f"},
    "MDVP:Fhi(Hz)": {"label": "2. MDVP:Fhi(Hz)", "description": "Frekuensi fundamental vokal maksimum.", "default": 197.10, "format": "%.3f"},
    "MDVP:Flo(Hz)": {"label": "3. MDVP:Flo(Hz)", "description": "Frekuensi fundamental vokal minimum.", "default": 116.32, "format": "%.3f"},
    "MDVP:Jitter(%)": {"label": "4. MDVP:Jitter(%)", "description": "Ukuran variasi persentase frekuensi fundamental.", "default": 0.0062, "format": "%.5f"},
    "MDVP:Jitter(Abs)": {"label": "5. MDVP:Jitter(Abs)", "description": "Ukuran absolut variasi frekuensi fundamental.", "default": 0.000044, "format": "%.6f"},
    "MDVP:RAP": {"label": "6. MDVP:RAP", "description": "Relative Average Perturbation, ukuran kegagalan nada.", "default": 0.0033, "format": "%.5f"},
    "MDVP:PPQ": {"label": "7. MDVP:PPQ", "description": "Five-point Period Perturbation Quotient, ukuran kegagalan nada.", "default": 0.0034, "format": "%.5f"},
    "Jitter:DDP": {"label": "8. Jitter:DDP", "description": "Average absolute difference of differences between consecutive periods.", "default": 0.0099, "format": "%.5f"},
    "MDVP:Shimmer": {"label": "9. MDVP:Shimmer", "description": "Ukuran variasi amplitudo vokal.", "default": 0.0297, "format": "%.5f"},
    "MDVP:Shimmer(dB)": {"label": "10. MDVP:Shimmer(dB)", "description": "Variasi amplitudo vokal dalam desibel.", "default": 0.2823, "format": "%.3f"},
    "Shimmer:APQ3": {"label": "11. Shimmer:APQ3", "description": "Three-point Amplitude Perturbation Quotient.", "default": 0.0165, "format": "%.5f"},
    "Shimmer:APQ5": {"label": "12. Shimmer:APQ5", "description": "Five-point Amplitude Perturbation Quotient.", "default": 0.0179, "format": "%.5f"},
    "MDVP:APQ": {"label": "13. MDVP:APQ", "description": "Ukuran variasi amplitudo vokal terhadap amplitudo rata-rata.", "default": 0.0241, "format": "%.5f"},
    "Shimmer:DDA": {"label": "14. Shimmer:DDA", "description": "Average absolute difference between consecutive differences of the period of the speech signal.", "default": 0.0470, "format": "%.5f"},
    "NHR": {"label": "15. NHR", "description": "Noise-to-Harmonic Ratio, rasio antara noise dan komponen harmonik.", "default": 0.0248, "format": "%.5f"},
    "HNR": {"label": "16. HNR", "description": "Harmonic-to-Noise Ratio, rasio antara komponen harmonik dan noise.", "default": 21.89, "format": "%.3f"},
    "RPDE": {"label": "17. RPDE", "description": "Recurrence Period Density Entropy, ukuran kompleksitas dinamika sistem.", "default": 0.4985, "format": "%.6f"},
    "DFA": {"label": "18. DFA", "description": "Detrended Fluctuation Analysis, ukuran korelasi dalam sinyal suara.", "default": 0.7181, "format": "%.6f"},
    "spread1": {"label": "19. spread1", "description": "Ukuran nonlinier variasi frekuensi fundamental.", "default": -5.6844, "format": "%.6f"},
    "spread2": {"label": "20. spread2", "description": "Ukuran nonlinier variasi frekuensi fundamental (kedua).", "default": 0.2265, "format": "%.6f"},
    "D2": {"label": "21. D2", "description": "Correlation Dimension, ukuran kompleksitas atau fractalitas sinyal.", "default": 2.3818, "format": "%.6f"},
    "PPE": {"label": "22. PPE", "description": "Pitch Period Entropy, ukuran kompleksitas periodisitas nada.", "default": 0.2066, "format": "%.6f"},
}

# Urutan fitur sesuai saat melatih model
# Ini sangat penting! Pastikan urutan ini cocok dengan X.columns saat model dilatih.
feature_order = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
    'spread1', 'spread2', 'D2', 'PPE'
]

# --- Streamlit App Interface ---
st.set_page_config(page_title="Parkinson's Disease Prediction", page_icon="🧠", layout="wide")
st.title("Aplikasi Prediksi Penyakit Parkinson")
st.markdown("Masukkan parameter suara pasien untuk memprediksi apakah mereka mengidap penyakit Parkinson.")

# Dictionary to store user inputs
user_inputs = {}

# Divide features into two columns for vertical layout
col1, col2 = st.columns(2)

# Input fields for the first half of the features
with col1:
    st.header("Bagian 1 dari 2")
    for i in range(len(feature_order) // 2): # Iterate through the first half of features
        feature_name = feature_order[i]
        info = features_info[feature_name]
        user_inputs[feature_name] = st.number_input(
            f"{info['label']}",
            value=info['default'],
            format=info['format'],
            help=info['description'] # Use help for description
        )

# Input fields for the second half of the features
with col2:
    st.header("Bagian 2 dari 2")
    for i in range(len(feature_order) // 2, len(feature_order)): # Iterate through the second half
        feature_name = feature_order[i]
        info = features_info[feature_name]
        user_inputs[feature_name] = st.number_input(
            f"{info['label']}",
            value=info['default'],
            format=info['format'],
            help=info['description'] # Use help for description
        )

# Button for prediction
st.markdown("---") # Garis pemisah untuk visual
parkinsons_diagnosis = ''

if st.button("Dapatkan Hasil Tes Parkinson"):
    try:
        # Create a list of input values in the correct order
        input_data = [user_inputs[feature_name] for feature_name in feature_order]

        # Reshape the numpy array for prediction
        input_data_as_numpy_array = np.asarray(input_data)
        input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

        # Make prediction
        prediction = loaded_model.predict(input_data_reshaped)

        if prediction[0] == 0:
            parkinsons_diagnosis = "Orang tersebut **TIDAK** mengidap Penyakit Parkinson."
        else:
            parkinsons_diagnosis = "Orang tersebut **MENGIDAP** Penyakit Parkinson."
        st.success(parkinsons_diagnosis)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses input atau membuat prediksi: {e}")
        st.info("Pastikan semua nilai yang dimasukkan sudah benar dan sesuai.")

# Optional: Add information about the features in the sidebar
st.sidebar.header("Informasi Tambahan")
st.sidebar.markdown(
    """
    Dataset ini terdiri dari berbagai pengukuran suara yang diambil dari individu.
    Perubahan tertentu dalam pola suara, seperti fluktuasi frekuensi dan amplitudo,
    seringkali menjadi indikator awal penyakit Parkinson.
    """
)
st.sidebar.markdown("---")
st.sidebar.info("Aplikasi ini dibuat untuk tujuan edukasi dan demonstrasi. Jangan gunakan sebagai pengganti diagnosis medis profesional.")
