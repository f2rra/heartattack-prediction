# Laporan Proyek Machine Learning - Fathur Rahman Al Farizy

## Domain Proyek

Penyakit kardiovaskular (kardiovaskuler) merupakan penyebab utama kematian di seluruh dunia, dan serangan jantung (infark miokard) adalah salah satu manifestasi akut yang paling mengancam jiwa. Dampak serangan jantung tidak hanya terbatas pada angka kematian yang tinggi, tetapi juga morbiditas yang signifikan, penurunan kualitas hidup pasien, dan beban ekonomi yang besar bagi sistem kesehatan dan masyarakat secara keseluruhan. Identifikasi dini individu yang berisiko tinggi mengalami serangan jantung menjadi krusial untuk memungkinkan intervensi preventif yang tepat waktu dan efektif, yang pada akhirnya dapat mengurangi insiden, morbiditas, dan mortalitas akibat kondisi ini.

## Business Understanding

### Problem Statements

**1. Keterbatasan Metode Penilaian Risiko Tradisional**  
Metode konvensional untuk menilai risiko serangan jantung seringkali mengandalkan faktor risiko klinis tunggal atau kombinasi linier yang mungkin tidak sepenuhnya menangkap kompleksitas interaksi antara berbagai variabel risiko. Data pasien modern yang kaya dan multidimensional memerlukan pendekatan analisis yang lebih canggih untuk mengidentifikasi pola-pola risiko yang subtil dan non-linear.

**2. Potensi Suboptimalisasi Pemanfaatan Data Pasien**  
Sejumlah besar data pasien yang relevan, termasuk hasil pemeriksaan fisik dan tes laboratorium, seringkali belum dimanfaatkan secara optimal untuk tujuan prediksi risiko serangan jantung. Pemanfaatan teknik machine learning dapat membuka potensi untuk mengekstrak wawasan yang lebih mendalam dari data ini dan meningkatkan akurasi prediksi.

### Goals

**1. Pengembangan Model Klasifikasi berbasis Machine Learning**  
Tujuan utama proyek ini adalah untuk mengembangkan model klasifikasi machine learning yang mampu mengintegrasikan berbagai fitur pasien secara efektif dan akurat untuk memprediksi tingkat risiko serangan jantung. Model ini diharapkan dapat mengatasi keterbatasan metode tradisional dalam menangani kompleksitas data dan mengidentifikasi pola risiko yang non-linear.

**2. Implementasi Algoritma XGBoost untuk Prediksi yang Lebih Baik**  
Proyek ini bertujuan untuk mengimplementasikan algoritma XGBoost (Extreme Gradient Boosting), yang dikenal dengan kemampuannya dalam menangani data tabular dan mencapai kinerja prediksi yang tinggi. Dengan memanfaatkan XGBoost, diharapkan model yang dikembangkan dapat memberikan prediksi risiko serangan jantung yang lebih akurat dan andal dibandingkan dengan pendekatan konvensional.

## Data Understanding

Dataset yang digunakan dalam proyek ini adalah "Heart Attack Risk Assessment Dataset". Dataset ini dikumpulkan di Zheen Hospital di Erbil, Irak, dari Januari hingga Mei 2019 yang berisi catatan medis pasien dengan tujuan utama untuk mengklasifikasikan apakah seorang individu mengalami serangan jantung. Berdasarkan jumlah datanya, dataset ini terdiri dari 1319 baris dan 11 kolom. Setelah dilakukan pengecekan, tidak terdapat missing value dan data yang duplikat.

Dataset ini dapat diakses melalui : https://www.kaggle.com/datasets/fajobgiua/heart-attack-risk-assessment-dataset.

**Fitur-fitur pada Heart Attack Risk Assessment Dataset adalah sebagai berikut:**

- **Age:** Usia pasien dalam tahun. Individu yang lebih tua umumnya memiliki risiko lebih tinggi untuk masalah kardiovaskular.
- **Gender:** Jenis kelamin merupakan faktor yang relevan dalam penyakit jantung, dengan profil risiko yang bervariasi berdasarkan jenis kelamin dan usia. Indikator biner jenis kelamin pasien:
  - 1 = Male (Laki-laki)
  - 0 = Female (Perempuan)
- **Heart rate:** Detak jantung per menit. Detak jantung yang tidak normal dapat mengindikasikan tekanan jantung atau masalah kesehatan mendasar.
- **Systolic blood pressure:** Tekanan darah saat jantung berdetak (sistolik) dalam mmHg. Nilai sistolik yang tinggi merupakan faktor risiko signifikan untuk serangan jantung.
- **Diastolic blood pressure:** Tekanan darah di antara detak jantung (diastolik) dalam mmHg. Tekanan diastolik yang meningkat juga berkontribusi pada risiko kardiovaskular.
- **Blood sugar:** Kadar gula darah (glukosa) dalam mg/dL. Meskipun bukan kolom biner, nilai di atas 120 mg/dL umumnya mengindikasikan hiperglikemia, yang merupakan faktor risiko untuk kondisi jantung, terutama pada pasien diabetes.
- **CK-MB:** Tingkat enzim Creatine kinase-MB, sebuah biomarker jantung. Tingkat yang tinggi dapat mengindikasikan kerusakan otot jantung.
- **Troponin:** Sebuah protein yang dilepaskan ke dalam darah ketika otot jantung rusak. Tingkat yang tinggi sangat terkait dengan serangan jantung.
- **Result:** Klasifikasi kondisi pasien:
  - positive = Pasien mengalami serangan jantung
  - negative = Tidak terdeteksi serangan jantung
- **Risk_Level:** Penilaian kategorikal risiko serangan jantung pasien berdasarkan indikator klinis. Variabel ini menjadi target klasifikasi. Nilainya meliputi:
  - Low (Rendah)
  - Moderate (Sedang)
  - High (Tinggi)
- **Recommendation:** Saran medis atau gaya hidup berdasarkan kondisi dan tingkat risiko pasien. Rekomendasi umum meliputi:
  - Immediate medical attention (Perhatian medis segera) untuk kasus berisiko tinggi atau terkonfirmasi.
  - Monitor closely and consult doctor (Pantau dengan ketat dan konsultasikan dengan dokter) untuk kasus berisiko sedang.
  - Maintain healthy lifestyle (Pertahankan gaya hidup sehat) untuk kasus berisiko rendah atau sebagai tindakan pencegahan.

## Data Preparation

Pada bagian ini, dilakukan serangkaian teknik persiapan data untuk memastikan kualitas dan kesiapan data sebelum digunakan untuk pemodelan. Langkah-langkah persiapan data yang diterapkan adalah sebagai berikut:

**1. Penanganan Outliers:**  
Dilakukan pemeriksaan keberadaan nilai outliers dalam dataset menggunakan fungsi `sns.boxplot()` untuk melihat data-data yang berada di luar jangkauan _Inter Quartile Range (IQR)_. Setelah diperiksa, terdapat beberapa data outliers pada beberapa kolom seperti Age, Heart rate, Systolic blood pressure, Diastolic blood pressure, Blood sugar, CK-MB, dan Troponin. Untuk mengatasi outliers ini, saya menerapkan metode IQR dengan menghilangkan data-data yang berada di rentang < Q1-1,5\*IQR dan > Q3+1.5\*IQR. Untuk beberapa kasus yang distribusi datanya highly right skew seperti data Blood sugar, CK-MB, dan Troponin dilakukan pembersihan outliers berdasarkan distribusi data pada histogramnya. Data yang frekuensinya sangat sedikit atau tidak signifikan dibanding data lainnya akan dihapus pada fitur-fitur ini.

**2. Transformasi Variabel Target:**  
Variabel target asli yaitu "Risk_Level" yang berupa data kategorikal diubah menjadi data numerik dengan metode \_ordinal encoding\* dikarenakan nilai kategorinya memiliki bobot makna. Kategori "Low", "Moderate", dan "High" diubah menjadi nilai numerik 0, 1, dan 2.

**3. Reduksi Dimensi:**  
Reduksi dimensi dilakukan menggunakan metode Principal Component Analysis (PCA). PCA dipilih karena efektif dalam mengubah fitur berkorelasi menjadi fitur tidak berkorelasi (komponen utama) sambil mempertahankan sebanyak mungkin informasi penting dalam data. Adapun fitur yang direduksi adalah fitur `Systolic blood pressure` dan `Diastolic blood pressure` yang berkorelasi kuat (0.62). Fitur baru ini dinamakan sebagai `blood_pressure`.

**4. Seleksi Fitur:**  
Dilakukan seleksi fitur dengan menghilangkan kolom "Recommendation" dan "Result" karena untuk pemodelan klasifikasi risiko serangan jantung data yang menggunakan target variabel Risk_Level, fitur ini dapat diwakilkan. Misalnya pada fitur Recommendation yang menyesuaikan dengan Risk Level-nya.

**5. Pembagian Data:**  
Data dibagi menjadi data latih (train) dan data uji (test) dengan proporsi 80:20 menggunakan fungsi `train_test_split` dari `sklearn.model_selection`. Pembagian ini memastikan bahwa model dievaluasi pada data yang belum pernah dilihat sebelumnya.
Data latih kemudian dibagi lagi menjadi data latih dan data validasi dengan proporsi 80:20. Data validasi ini digunakan untuk optimasi hyperparameter.
Hasil dari proses pembagian data ini adalah 20% data digunakan sebagai data testing untuk evaluasi model pada data-data baru yang belum pernah dilihat. 20% data digunakan sebagai data validasi yang berperan sebagai acuan akurasi model dan proses hyperparameter tuning. Lalu 60% data digunakan sebagai data training untuk melatih model.

**6. Feature Scaling:**  
Dilakukan penskalaan fitur numerik menggunakan `StandardScaler` dari `sklearn.preprocessing`. Proses ini mentransformasi fitur-fitur numerik sehingga memiliki rata-rata 0 dan deviasi standar 1. Penskalaan penting untuk mencegah fitur dengan skala besar mendominasi fitur dengan skala kecil selama pelatihan model.

## Modeling

**1. Pemilihan Algoritma: XGBoost**  
XGBoost (Extreme Gradient Boosting) adalah algoritma boosting tree yang telah terbukti efektif dalam berbagai tugas klasifikasi dan regresi. XGBoost bekerja dengan cara menggabungkan banyak _weak learner_ (pohon keputusan) secara aditif. XGBoost membangun pohon secara sekuensial, di mana setiap pohon berusaha memperbaiki kesalahan prediksi dari pohon sebelumnya. XGBoost menggunakan gradient boosting, yaitu meminimalkan loss function dengan menambahkan pohon yang paling curam penurunannya (steepest descent) pada setiap iterasi.

**Kelebihan XGBoost:**

- **Kinerja tinggi:** Mampu mencapai akurasi yang sangat baik.
- **Efisiensi komputasi:** Implementasi yang dioptimalkan untuk kecepatan.
- **Penanganan missing values bawaan.**
- **Regularisasi untuk mencegah overfitting.**
- **Interpretasi model:** Menyediakan fitur importance.

**Kekurangan XGBoost:**

- **Sensitif terhadap hyperparameter:** Membutuhkan tuning yang cermat.
- **Dapat overfit jika tidak diatur dengan baik.**
- **Kurva belajar mungkin lebih lambat dibandingkan beberapa algoritma lain.**

**2. Tahapan Pemodelan:**

- **Inisialisasi Model:** Model XGBoost Classifier (XGBClassifier) diinisialisasi dengan parameter dasar yaitu 'objective': 'multi:softmax' (untuk menjelaskan targetnya berupa multikelas); 'num_class': 3 (jumlah kelas sebanyak 3); num_boost_round=100 (jumlah total pohon yang akan dibangun); early_stopping_rounds: Parameter ini mengaktifkan mekanisme early stopping..
- **Pelatihan Model:** Model dilatih menggunakan data latih dtrain [xgb.DMatrix(X_train, label=y_train)] dengan beberapa tambahan parameter seperti early_stopping_rounds=10 (membatasi pelatihan sebanyak 10 kali setelah tidak ada peningkatan pada metrik evaluasi).
- **Prediksi pada Data Uji:** Model yang telah dilatih digunakan untuk membuat prediksi pada data uji dtest [xgb.DMatrix(X_test)].

**3. Hyperparameter Tuning:**  
Hyperparameter tuning dilakukan untuk mencari kombinasi parameter terbaik yang menghasilkan kinerja model yang optimal. Metode _GridSearchCV_ dengan stratified k-fold cross-validation digunakan untuk melakukan pencarian grid.

- Metode: _GridSearchCV_ secara sistematis mencoba semua kombinasi hyperparameter yang ditentukan dalam param_grid dan mengevaluasi kinerja model menggunakan stratified k-fold cross-validation dengan 5 folds. Stratified k-fold memastikan bahwa proporsi kelas di setiap fold serupa dengan proporsi di dataset asli, yang penting untuk menangani potensi class imbalance. Proses tuning ini menghasilkan total 135 fits (5 folds x 27 kandidat parameter).
- Hyperparameter yang Di-tune:
  - `n_estimators`: Jumlah pohon dalam model XGBoost. Rentang nilai yang diuji: [100, 200, 300].
  - `learning_rate`: Laju pembelajaran, yang mengontrol seberapa besar setiap pohon berkontribusi pada prediksi akhir. Rentang nilai yang diuji: [0.01, 0.1, 0.2].
  - `max_depth`: Kedalaman maksimum setiap pohon. Rentang nilai yang diuji: [3, 4, 5].
- Metrik Evaluasi: Akurasi digunakan sebagai metrik evaluasi untuk memilih parameter terbaik.
- Parameter Terbaik: Setelah proses tuning, diperoleh parameter terbaik sebagai berikut:

  - `n_estimators`: 100
  - `learning_rate`: 0.2
  - `max_depth`: 3

  Kombinasi parameter ini menghasilkan skor akurasi cross-validation terbaik sebesar 0.9657. `max_depth` yang rendah (3) menunjukkan bahwa model menggunakan pohon yang relatif sederhana, yang dapat membantu mencegah _overfitting_. `learning_rate` yang moderat (0.2) memungkinkan model untuk belajar dengan cukup cepat. Jumlah pohon yang optimal (100) memberikan keseimbangan antara kinerja dan efisiensi komputasi.

Model XGBoost kemudian dilatih ulang dengan parameter terbaik ini pada seluruh data pelatihan untuk mendapatkan model akhir. Model akhir ini dievaluasi pada set pengujian, menghasilkan akurasi sebesar 0.9702.

## Evaluation

**1. Metrik Evaluasi yang Digunakan:**

- **Akurasi (Accuracy):**

  - Penjelasan: Akurasi adalah metrik yang mengukur proporsi prediksi yang benar dari total prediksi. Ini memberikan gambaran keseluruhan seberapa sering model benar dalam memprediksi tingkat risiko serangan jantung.
  - Formula: Akurasi = (Jumlah Prediksi Benar) / (Total Jumlah Prediksi)
  - Cara Kerja: Akurasi dihitung dengan membagi jumlah total prediksi yang benar (yaitu, jumlah kasus di mana model memprediksi tingkat risiko yang benar) dengan jumlah total prediksi (yaitu, jumlah total pasien dalam set pengujian).

  **Presisi (Precision):**

  - Penjelasan: Presisi adalah metrik yang mengukur proporsi prediksi positif yang benar dari total prediksi positif. Untuk setiap kelas risiko (Low, Moderate, High), presisi mengukur seberapa sering model benar dalam memprediksi bahwa seorang pasien termasuk dalam kelas tersebut.
  - Formula: Presisi = (Jumlah Prediksi Positif Benar) / (Total Jumlah Prediksi Positif)
  - Cara Kerja: Untuk setiap kelas, presisi dihitung dengan membagi jumlah kasus di mana model dengan benar memprediksi kelas tersebut dengan jumlah total kasus yang diprediksi oleh model sebagai kelas tersebut.

  **Recall (Sensitivity atau True Positive Rate):**

  - Penjelasan: Recall adalah metrik yang mengukur proporsi kasus positif yang benar yang berhasil diidentifikasi oleh model dari total kasus positif yang sebenarnya. Untuk setiap kelas risiko, recall mengukur seberapa baik model dalam menemukan semua pasien yang sebenarnya termasuk dalam kelas tersebut.
  - Formula: Recall = (Jumlah Prediksi Positif Benar) / (Total Jumlah Kasus Positif Sebenarnya)
  - Cara Kerja: Untuk setiap kelas, recall dihitung dengan membagi jumlah kasus di mana model dengan benar memprediksi kelas tersebut dengan jumlah total kasus yang sebenarnya termasuk dalam kelas tersebut.

  **F1-score:**

  - Penjelasan: F1-score adalah rata-rata harmonik dari presisi dan recall. Metrik ini memberikan keseimbangan antara presisi dan recall, yang berguna ketika ada ketidakseimbangan kelas.
  - Formula: F1-score = 2 _ (Presisi _ Recall) / (Presisi + Recall)
  - Cara Kerja: F1-score dihitung untuk setiap kelas menggunakan nilai presisi dan recall untuk kelas tersebut.

  **Confusion Matrix:**

  - Penjelasan: Matriks konfusi adalah tabel yang merangkum hasil prediksi model klasifikasi. Ini menunjukkan jumlah prediksi yang benar dan salah untuk setiap kombinasi kelas aktual dan kelas prediksi.
  - Cara Kerja: Baris matriks mewakili kelas aktual, kolom mewakili kelas prediksi, dan setiap sel dalam matriks menunjukkan jumlah prediksi yang termasuk dalam kombinasi kelas tersebut.

**2. Hasil Proyek Berdasarkan Metrik Evaluasi:**

- Akurasi: Model XGBoost yang telah di-tune mencapai akurasi sebesar 97% pada data uji. Ini berarti model benar dalam memprediksi tingkat risiko serangan jantung untuk 97% pasien dalam set pengujian.
- Laporan Klasifikasi:
  - Presisi:
    - Kelas Low: 95%
    - Kelas Moderate: 97%
    - Kelas High: 99%
  - Recall:
    - Kelas Low: 96%
    - Kelas Moderate: 97%
    - Kelas High: 97%
  - F1-score:
    - Kelas Low: 95%
    - Kelas Moderate: 97%
    - Kelas High: 98%
  - Interpretasi:
    - Model menunjukkan kinerja klasifikasi yang sangat baik di semua kelas risiko (Low, Moderate, dan High) berdasarkan metrik presisi, recall, dan F1-score yang tinggi.
    - Kelas Low: Dengan presisi 95% dan recall 96%, model secara akurat mengidentifikasi pasien dengan risiko rendah dengan kesalahan yang minimal, baik dalam memprediksi pasien sebagai risiko rendah (presisi) maupun dalam menemukan semua pasien yang sebenarnya berisiko rendah (recall).
    - Kelas Moderate: Model memiliki kinerja yang sangat baik untuk kelas risiko sedang, dengan presisi 97% dan recall 97%. Ini menunjukkan bahwa model sangat andal dalam mengklasifikasikan pasien dengan risiko sedang.
    - Kelas High: Model menunjukkan kinerja yang luar biasa untuk kelas risiko tinggi. Dengan presisi 99%, hampir semua pasien yang diprediksi sebagai risiko tinggi oleh model memang benar-benar berisiko tinggi. Recall sebesar 97% juga sangat baik, yang berarti model berhasil mengidentifikasi sebagian besar pasien yang sebenarnya berisiko tinggi. F1-score sebesar 98% mengkonfirmasi keseimbangan yang baik antara presisi dan recall untuk kelas ini.
- Confusion Matrix:
  - Kelas Low: 52 prediksi benar, 2 prediksi salah (1 Moderate, 1 High)
  - Kelas Moderate: 38 prediksi benar, 1 prediksi salah (1 Low)
  - Kelas High: 73 prediksi benar, 2 prediksi salah (2 Low)
  - Interpretasi:
    - Model cenderung melakukan kesalahan dengan memprediksi beberapa kasus Low sebagai Moderate atau High, dan beberapa kasus High sebagai Low.
    - Jumlah kesalahan prediksi secara keseluruhan relatif kecil, yang mengkonfirmasi kinerja model yang baik.

**3. Kesimpulan:**

Hasil evaluasi model XGBoost menunjukkan keberhasilan dalam menjawab problem statement yang diidentifikasi di awal. Akurasi model yang tinggi (97%) secara signifikan mengatasi keterbatasan metode penilaian risiko tradisional yang seringkali gagal menangkap kompleksitas interaksi antar faktor risiko. Model ini mampu memanfaatkan data pasien yang kaya dan multidimensional untuk memberikan prediksi yang lebih akurat, yang sejalan dengan goal pengembangan model klasifikasi yang efektif. Selain itu, kinerja model yang baik pada berbagai metrik (presisi, recall, F1-score) menunjukkan bahwa model ini mampu mengoptimalkan pemanfaatan data pasien, menjawab problem statement kedua terkait potensi suboptimalisasi data.

Implementasi model XGBoost sebagai solusi yang diusulkan memiliki dampak positif yang signifikan. Prediksi risiko yang lebih akurat memungkinkan tenaga medis untuk mengidentifikasi pasien berisiko tinggi dengan lebih tepat, memfasilitasi intervensi preventif yang lebih tepat waktu dan efektif. Hal ini berpotensi mengurangi insiden serangan jantung, morbiditas, dan mortalitas, serta meningkatkan efisiensi alokasi sumber daya dalam sistem kesehatan. Dengan demikian, model ini berkontribusi pada pencapaian goal keseluruhan proyek, yaitu meningkatkan manajemen risiko serangan jantung melalui pemanfaatan machine learning.
