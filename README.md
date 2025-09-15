# Aircraft Passenger Satisfaction Prediction

## Repository Outline

1. README.md - Penjelasan gambaran umum project
2. notebook.ipynb - Notebook yang berisi pembuatan model prediksi
3. notebook_inference.ipynb - Notebook yang berisi inferensi data pada environment di luar permodelan

## Problem Background

Industri penerbangan sangat bergantung pada tingkat kepuasan penumpang sebagai indikator kualitas layanan dan daya saing. Namun, kepuasan penumpang dipengaruhi oleh banyak faktor, seperti ketepatan waktu, kenyamanan, pelayanan kabin, hingga proses check-in. Mengidentifikasi dan memprediksi kepuasan pelanggan secara manual membutuhkan waktu dan sumber daya yang besar.
Proyek ini bertujuan untuk membangun model prediktif berbasis machine learning yang dapat secara otomatis memprediksi tingkat kepuasan penumpang berdasarkan data survei dan informasi penerbangan. Dengan prediksi ini, maskapai dapat dengan cepat mengambil langkah perbaikan pada aspek layanan yang paling berpengaruh terhadap pengalaman pelanggan.

## Project Output

- **Model Machine Learning Terlatih**
  - Algoritma: Random Forest Classifier dengan hyperparameter hasil tuning.
  - Akurasi pada data uji: 95.86%.
  - Pipeline lengkap yang mencakup feature engineering, preprocessing, dan model.
- **File Model dalam Format `.pkl`**
  - Disimpan menggunakan cloudpickle agar bisa di-load di luar lingkungan training.
  - Siap digunakan untuk inference pada platform deployment seperti Hugging Face atau API service.
- **Notebook Inference**
  - Contoh implementasi untuk memuat model `.pkl` dan melakukan prediksi pada data mentah.
- **Dokumentasi Notebook Utama**
  - Meliputi seluruh proses mulai dari penggabungan dataset, pembersihan data, feature engineering, model training & tuning, evaluasi, hingga penyimpanan model.
- **Deployment**
  - Model di-*deploy* menggunakan **Hugging Face Spaces** dengan Streamlit sebagai interface.
  - Memungkinkan pengguna melakukan prediksi secara interaktif dengan mengunggah file CSV atau mengisi input secara manual.
  - Link deployment: *[Aircraft Passenger Satisfaction Prediction](akan diisi setelah proses deploy selesai)*

## Data

Dataset bersumber dari [Kaggle - Airline Passenger Satisfaction](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction/data).  
Karakteristik data:
- Jumlah data gabungan: ± 129.880 baris (train + test).
- Jumlah kolom: 25 fitur termasuk label `satisfaction`.
- Variabel terdiri dari data kategorikal, ordinal, dan numerik.
- Terdapat missing values pada beberapa kolom delay.
- Label target: `Satisfied` dan `Neutral or Dissatisfied`.

## Method

Metode yang digunakan adalah supervised learning untuk klasifikasi biner.  
Tahapan yang dilakukan meliputi:
1. **Data Preparation**: penggabungan dataset train dan test, pembersihan missing values, dan pengolahan kolom ordinal.
2. **Feature Engineering**: pembuatan variabel baru, binning, dan encoding.
3. **Model Training**: evaluasi baseline model menggunakan KNN, SVM, Decision Tree, Random Forest, dan Gradient Boosting.
4. **Hyperparameter Tuning**: optimasi parameter pada Random Forest menggunakan RandomizedSearchCV (5-fold cross-validation).
5. **Pipeline Development**: menyatukan seluruh tahapan preprocessing dan model ke dalam pipeline untuk memudahkan deployment.
6. **Model Saving**: penyimpanan model dengan cloudpickle untuk kompatibilitas di luar environment training.
7. **Inference**: pengujian model pada data mentah di notebook terpisah.

## Stacks

- **Bahasa Pemrograman**: Python 3
- **Environment**: VSCode, Conda
- **Library Utama**:
  - Data handling: pandas, numpy
  - Visualization: matplotlib, seaborn
  - Machine Learning: scikit-learn
  - Model persistence: cloudpickle, joblib

## Reference
`Bagian ini berisi link pendukung seperti referensi, dashboard, atau deployment`

---

**Referensi tambahan:**
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)
