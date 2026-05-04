# Wine Quality Classification - UTS Data Mining

## Pendahuluan
Kualitas anggur merupakan salah satu faktor kritis dalam industri minuman yang secara tradisional dinilai melalui proses pengujian organoleptik oleh para ahli (wine tasters). Proses penilaian tersebut bersifat subjektif, membutuhkan keahlian khusus, dan tidak efisien untuk dilakukan dalam skala besar. Seiring berkembangnya teknologi analisis kimiawi, kini dimungkinkan untuk mengukur berbagai senyawa kimia dalam anggur secara objektif dan terstandarisasi.

Dalam konteks tersebut, pendekatan machine learning menawarkan solusi yang lebih skalabel dan objektif. Dengan memanfaatkan data fitur kimiawi yang terukur, sebuah model klasifikasi dapat dilatih untuk mempelajari pola hubungan antara komposisi kimiawi anggur dan skor kualitasnya, sehingga penilaian kualitas dapat dilakukan secara otomatis tanpa bergantung sepenuhnya pada penilaian manusia.

Proyek ini merupakan bagian dari Ujian Tengah Semester (UTS) mata kuliah Data Mining. Tujuan utamanya adalah membangun model klasifikasi yang mampu memprediksi nilai kualitas anggur berdasarkan fitur-fitur kimiawi yang tersedia. Model dilatih menggunakan data training berlabel, kemudian diterapkan untuk memprediksi kualitas pada data testing yang tidak memiliki label.

## Informasi Repositori
Repositori ini berisi hasil pengerjaan UTS Data Mining dengan studi kasus prediksi kualitas anggur menggunakan pendekatan machine learning. Model dibangun dari data training, dievaluasi, kemudian digunakan untuk memprediksi kualitas anggur pada data testing yang tidak memiliki label. Repositori ini berisi seluruh berkas yang dibutuhkan untuk mereproduksi proses analisis dari awal hingga menghasilkan file prediksi. Notebook disusun secara sistematis dengan penjelasan dan interpretasi di setiap tahapan sehingga dapat dipahami tanpa perlu menjalankan kode terlebih dahulu.

---

## Struktur Repositori

```
.
├── wine_quality_classification.ipynb   # Notebook utama beserta dokumentasi dan interpretasi
├── data_training.csv                   # Dataset pelatihan (857 sampel)
├── data_testing.csv                    # Dataset pengujian (286 sampel, tanpa label quality)
├── hasilprediksi_NIM.csv               # Hasil prediksi kualitas anggur (Id;Quality)
└── README.md                           # Dokumentasi proyek
```

---

## Deskripsi Dataset

Dataset yang digunakan adalah Wine Quality Dataset yang bersumber dari UCI Machine Learning Repository. Dataset ini berisi hasil pengukuran senyawa kimiawi dari sampel anggur merah (red wine) dan anggur putih (white wine) beserta skor kualitas yang diberikan oleh panel penguji bersertifikat. Variabel target yang diprediksi adalah `quality`, yaitu nilai kualitas anggur pada skala 0 hingga 10.

Data training yang digunakan dalam proyek ini terdiri dari 857 sampel dengan distribusi kelas yang tidak seimbang, di mana sebagian besar sampel berada pada kelas kualitas menengah (quality 5 dan 6). Data testing terdiri dari 286 sampel tanpa label quality yang harus diprediksi.

### Fitur yang Digunakan

| Fitur | Deskripsi | Penjelasan |
|---|---|---|
| fixed acidity | Kadar asam tetap (g/dm3) | Kadar asam tetap, umumnya berupa asam tartarat. Asam ini tidak mudah menguap dan memberikan rasa segar (tart) pada anggur. Kadar yang terlalu tinggi dapat membuat rasa terlalu asam dan tajam. |
| volatile acidity | Kadar asam volatil (g/dm3) | Kadar asam volatil, terutama asam asetat. Kadar yang tinggi dapat menyebabkan aroma dan rasa seperti cuka yang tidak diinginkan, sehingga berpengaruh negatif terhadap kualitas anggur. |
| citric acid | Kadar asam sitrat (g/dm3) | Kadar asam sitrat. Dalam jumlah kecil, asam sitrat dapat menambah kesegaran dan kompleksitas rasa. Kadarnya pada anggur umumnya sangat rendah karena sebagian besar difermentasi oleh bakteri. |
| residual sugar | Kadar gula sisa setelah fermentasi (g/dm3) | Kadar gula yang tersisa setelah proses fermentasi selesai. Nilai rendah menandakan anggur kering (dry wine), sedangkan nilai tinggi menandakan anggur manis (sweet wine). Kadar di atas 45 g/dm3 dianggap sangat manis. |
| chlorides | Kadar garam klorida (g/dm3) | Kadar garam (natrium klorida) dalam anggur. Kadar klorida yang terlalu tinggi dapat memberikan rasa asin yang tidak diinginkan dan mengindikasikan kontaminasi atau proses produksi yang kurang baik. |
| free sulfur dioxide | SO2 bebas (mg/dm3) | Kadar SO2 dalam bentuk bebas (tidak terikat). SO2 bebas berfungsi sebagai antimikroba dan antioksidan untuk menjaga kesegaran anggur. Kadar yang tepat diperlukan agar tidak merusak aroma dan rasa. |
| total sulfur dioxide | Total SO2 bebas dan terikat (mg/dm3) | Total keseluruhan SO2, mencakup bentuk bebas dan yang sudah terikat dengan senyawa lain. Regulasi pangan di berbagai negara membatasi kadar total SO2 pada produk anggur komersial. |
| density | Kepadatan anggur (g/cm3) | Kepadatan cairan anggur. Kepadatan berkaitan erat dengan kadar gula dan alkohol; anggur dengan kadar gula tinggi umumnya memiliki densitas lebih tinggi, sedangkan kadar alkohol tinggi menurunkan densitas. |
| pH | Tingkat keasaman | Ukuran keasaman atau kebasaan pada skala 0-14. Sebagian besar anggur memiliki pH antara 3 dan 4. Nilai pH yang lebih rendah berarti anggur lebih asam, yang memengaruhi stabilitas, warna, dan rasa. |
| sulphates | Kadar sulfat sebagai aditif antimikroba (g/dm3) | Kadar kalium sulfat, yang berfungsi sebagai aditif antimikroba dan antioksidan. Sulphates berkontribusi pada kadar SO2 dalam anggur dan secara umum berpengaruh positif terhadap kualitas dalam batas wajar. |
| alcohol | Kadar alkohol (% volume) | Persentase kadar alkohol dalam anggur. Alkohol dihasilkan dari proses fermentasi gula oleh ragi. Kadar alkohol yang lebih tinggi umumnya berkorelasi positif dengan kualitas anggur yang lebih baik. |
| quality | Nilai kualitas anggur - variabel target (0-10) | Skor kualitas anggur berdasarkan penilaian panel ahli (wine tasters) bersertifikat. Nilai 0 merupakan kualitas terburuk dan 10 adalah kualitas terbaik. Variabel ini merupakan target prediksi. |

---

## Alur Pengerjaan

### Langkah 1: Persiapan Data

Dataset training dan testing dimuat menggunakan pandas. Data training terdiri dari 857 baris dan 13 kolom (11 fitur, 1 kolom Id, 1 kolom target quality), sedangkan data testing terdiri dari 286 baris dan 12 kolom tanpa kolom quality.

### Langkah 2: Eksplorasi Data (EDA)

Sebelum membangun model, dilakukan eksplorasi untuk memahami karakteristik data:

- Statistik deskriptif menunjukkan perbedaan skala yang signifikan antar fitur. Misalnya, total sulfur dioxide memiliki nilai maksimum 289, sedangkan density hanya berkisar di 0.990 hingga 1.004.
- Distribusi kelas quality sangat tidak seimbang. Kelas quality 5 mendominasi dengan 362 sampel (42.2%), diikuti quality 6 dengan 341 sampel (39.8%). Kelas ekstrem seperti quality 3 hanya memiliki 6 sampel dan quality 8 hanya 13 sampel.
- Heatmap korelasi menunjukkan bahwa alcohol memiliki korelasi positif tertinggi dengan quality (r = 0.44), sedangkan volatile acidity berkorelasi negatif terkuat (r = -0.39). Ditemukan juga multikolinearitas antara free sulfur dioxide dan total sulfur dioxide (r = 0.72), namun hal ini tidak berpengaruh signifikan pada model Random Forest.

### Langkah 3: Pembersihan Data

Dilakukan pengecekan terhadap dua hal utama:

- Missing values: tidak ditemukan nilai kosong pada seluruh kolom di kedua dataset (total 0 missing values). Tidak diperlukan imputasi maupun penghapusan baris.
- Duplikat: tidak ditemukan baris duplikat pada data training maupun data testing.

Data dinyatakan bersih dan siap untuk diproses.

### Langkah 4: Feature Scaling

StandardScaler diterapkan untuk menormalisasi seluruh 11 fitur sehingga rata-rata menjadi 0 dan standar deviasi menjadi 1. Proses `fit` hanya dilakukan pada data training untuk menghindari data leakage. Data testing hanya di-`transform` menggunakan parameter yang sudah dipelajari dari data training, bukan di-fit ulang.

### Langkah 5: Pembuatan dan Evaluasi Model

Model yang digunakan adalah Random Forest Classifier dengan parameter:
- `n_estimators = 500` (jumlah pohon keputusan)
- `random_state = 42` (untuk reprodusibilitas hasil)

Evaluasi dilakukan menggunakan 5-Fold Stratified Cross Validation, di mana Stratified memastikan proporsi kelas tetap terjaga di setiap fold. Hasil yang diperoleh:

- Fold 1: 57.56%
- Fold 2: 65.12%
- Fold 3: 63.74%
- Fold 4: 66.08%
- Fold 5: 67.84%
- Rata-rata akurasi: 64.07% dengan standar deviasi 3.52%

Setelah cross validation, model dilatih ulang pada seluruh data training. Akurasi in-sample mencapai 100% dengan precision, recall, dan F1-score sempurna (1.00) untuk semua kelas. Ini adalah perilaku umum Random Forest karena model mampu menghafal seluruh data training. Estimasi performa generalisasi yang lebih realistis tetap mengacu pada hasil cross validation, yaitu 64.07%.

Analisis feature importance menunjukkan tiga fitur paling berpengaruh:
1. alcohol: 0.1418
2. sulphates: 0.1172
3. volatile acidity: 0.1109

Fitur dengan kontribusi terkecil adalah free sulfur dioxide (0.0664) dan residual sugar (0.0674). Hasil ini konsisten dengan temuan pada tahap EDA.

### Langkah 6: Penyimpanan Model

Model dan scaler yang sudah dilatih disimpan menggunakan joblib agar dapat di-load ulang tanpa perlu melatih ulang dari awal.

### Langkah 7: Prediksi Data Testing

Model yang disimpan di-load kembali dan digunakan untuk memprediksi quality pada 286 data testing. Distribusi hasil prediksi:

- Quality 5: 132 sampel (46.2%)
- Quality 6: 124 sampel (43.4%)
- Quality 7: 30 sampel (10.5%)

Tidak ada prediksi untuk kelas 3, 4, maupun 8. Distribusi ini konsisten dengan pola pada data training di mana kelas 5 dan 6 mendominasi.

Hasil prediksi kemudian disusun sesuai urutan template yang diberikan dan disimpan dalam file CSV dengan separator titik koma (;) yang hanya berisi dua kolom: Id dan Quality.

---

## Hasil Evaluasi

| Metrik | Nilai |
|---|---|
| Cross Validation Accuracy (5-Fold) | 64.07% |
| Standar Deviasi CV | 3.52% |
| Training Accuracy (in-sample) | 100% |
| Algoritma | Random Forest Classifier |
| Jumlah Pohon | 500 |
| Feature Scaling | StandardScaler |
| Fitur Terpenting | alcohol (0.1418), sulphates (0.1172), volatile acidity (0.1109) |

---

## Format Output

File hasil prediksi menggunakan separator titik koma (`;`) dan hanya memuat dua kolom sesuai ketentuan:

```
Id;Quality
222;5
1514;6
417;5
...
```

Total baris: 286 (sesuai jumlah data testing).

---

## Cara Menjalankan Notebook

**Google Colab (direkomendasikan):**

1. Buka [Google Colab](https://colab.research.google.com/)
2. Upload file `wine_quality_classification.ipynb`
3. Upload juga file `data_training.csv`, `data_testing.csv`, dan `hasilprediksi_010.csv` ke sesi Colab
4. Jalankan semua cell secara berurutan dari atas ke bawah (`Runtime > Run all`)
5. File hasil prediksi akan otomatis terdownload setelah cell terakhir dijalankan

**Catatan:** Sebelum menjalankan, ubah nilai `output_filename` di cell terakhir dengan mengganti `NIM` dengan 3 digit terakhir NIM saya yaitu 010.

---

## Library yang Digunakan

| Library | Kegunaan |
|---|---|
| pandas | Manipulasi dan analisis data |
| numpy | Komputasi numerik |
| matplotlib | Visualisasi data |
| seaborn | Visualisasi statistik |
| scikit-learn | Pembuatan dan evaluasi model machine learning |
| joblib | Penyimpanan dan loading model |
