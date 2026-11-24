# 🩺 Klasifikasi Kanker Payudara — Breast Cancer Wisconsin

Membandingkan tiga algoritma dan dua metode seleksi fitur untuk memisahkan tumor **malignant** dan **benign**, dengan prioritas pada seminimal mungkin kasus kanker yang terlewat.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Dataset](https://img.shields.io/badge/Dataset-Wisconsin%20Diagnostic-4B8BBE)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

**📓 Notebook:** [Colab](https://colab.research.google.com/drive/1QI4iF19RXalu6R0tu_z7A0wOsIzqSuc_) · **📦 Dataset:** [Kaggle — Breast Cancer Wisconsin](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

---

## 📌 Ringkasan Eksekutif

| Pertanyaan | Jawaban |
|---|---|
| Apa masalahnya? | Memisahkan tumor **malignant** (37,3%) dari **benign** (62,7%) dari 30 fitur pengukuran sel |
| Model terbaik? | **RFE + Logistic Regression** — F1 **0,9639**, hanya **2 dari 42** kanker terlewat |
| Seberapa andal? | ROC-AUC **0,9974**, PR-AUC **0,9957** |
| Apa yang paling penting? | Ukuran dan bentuk sel: `concave points`, `perimeter`, `radius`, `area` |
| Apa risikonya? | Dataset hanya 569 pasien — terlalu kecil untuk klaim "siap dipakai di rumah sakit" |

> **Insight utama:** dalam diagnosis kanker, **akurasi bukan metrik yang paling penting — false negative adalah segalanya**. Model terbaik di sini mencapai akurasi 97,37%, tapi empat model lain juga mencapai angka serupa. Yang membedakan adalah **hanya 2 kasus kanker terlewat** (recall 0,9524), versus 5 kasus pada model terlemah. Pada penyakit yang mengancam nyawa, selisih 3 pasien itulah alasan sebenarnya sebuah model dipilih.

---

## 🎯 Business Understanding

### Latar Belakang

Kanker payudara adalah salah satu kanker paling umum di dunia dengan lebih dari 2 juta kasus baru setiap tahunnya [1]. Deteksi dini sangat menentukan tingkat kelangsungan hidup pasien, dan keputusan diagnosis yang salah membawa konsekuensi berat ke dua arah:

- **False negative** (kanker diprediksi jinak) → pasien tidak mendapat penanganan, penyakit berkembang.
- **False positive** (jinak diprediksi kanker) → tindakan medis invasif yang tidak perlu, beban psikologis, dan biaya.

Dataset Breast Cancer Wisconsin menyediakan pengukuran numerik dari citra biopsi (radius, tekstur, kekompakan, cekungan, dan sebagainya) yang dapat dipakai untuk melatih model klasifikasi.

### Problem Statement

1. Bagaimana mengklasifikasikan tumor payudara sebagai malignant atau benign berdasarkan fitur pengukuran tumor dengan akurasi tinggi?
2. Algoritma machine learning mana yang memberikan performa terbaik untuk tugas ini?
3. Bagaimana memastikan model yang dibangun layak dipercaya untuk mendukung keputusan medis?

### Goals

- Membangun model dengan akurasi minimal **95%**.
- Membandingkan **tiga algoritma** (Logistic Regression, Random Forest, SVM) dengan **dua metode seleksi fitur** (SelectKBest, RFE) — total 6 kombinasi.
- Menyediakan evaluasi terukur dengan Accuracy, Precision, Recall, F1, ROC-AUC, dan PR-AUC.

### Metrik Keberhasilan

- **Metrik bisnis:** jumlah kasus malignant yang terlewat (**false negative**) — seminimal mungkin.
- **Metrik model:** **Recall** pada kelas malignant diprioritaskan, dilengkapi F1 sebagai penyeimbang dan ROC-AUC/PR-AUC untuk menilai kualitas pemisahan.

### Cakupan

**Termasuk:** EDA, preprocessing, seleksi fitur, perbandingan 6 kombinasi model, evaluasi komprehensif, rekomendasi.
**Tidak termasuk:** validasi eksternal pada rumah sakit lain, uji klinis prospektif, integrasi PACS/SIMRS, analisis citra mentah (data sudah berupa fitur terukur).

---

## 🗂️ Dataset

| Atribut | Keterangan |
|---|---|
| **Sumber** | [Kaggle — Breast Cancer Wisconsin (Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) |
| **Jumlah baris** | **569** pasien |
| **Jumlah kolom** | 33 (30 fitur + `id` + `diagnosis` + 1 kolom kosong) |
| **Missing value** | Tidak ada pada fitur utama; `Unnamed: 32` kosong sepenuhnya (562 nilai kosong) |
| **Target** | `diagnosis` — `M` (malignant) / `B` (benign) |

**Distribusi target:**

| Kelas | Jumlah | Proporsi |
|---|---:|---:|
| Benign (0) | 357 | **62,7%** |
| **Malignant (1)** | **212** | **37,3%** |

Distribusinya tidak terlalu timpang, sehingga tidak diperlukan resampling — berbeda dengan kasus churn atau dropout yang kelasnya sangat minoritas.

**Kelompok fitur (30 fitur numerik, 3 kelompok × 10 pengukuran):**

| Kelompok | Arti | Contoh |
|---|---|---|
| `_mean` | Rata-rata tiap pengukuran | `radius_mean`, `texture_mean`, `concavity_mean` |
| `_se` | Standard error (variasi pengukuran) | `radius_se`, `area_se` |
| `_worst` | Nilai terburuk (maksimum dari 3 sel terbesar) | `radius_worst`, `concave points_worst` |

**Pengukuran yang tersedia:** radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension.

**Kolom yang dibuang:**

- `id` — pengenal unik, tidak punya makna prediktif
- `Unnamed: 32` — kosong sepenuhnya

---

## 🔬 Metodologi

```
Data Loading → EDA → Cleaning → Encoding → Split 80:20 → Scaling → Feature Selection → Modeling → Evaluasi
```

1. **Data Loading** — 569 baris × 33 kolom.
2. **EDA** — struktur data, statistik deskriptif, distribusi target, heatmap korelasi, korelasi fitur terhadap target.
3. **Cleaning** — drop `id` dan `Unnamed: 32`.
4. **Encoding** — target `M`/`B` → 1/0.
5. **Split** — 80:20 **stratified** (`random_state=42`) → 455 train / 114 test.
6. **Scaling** — `StandardScaler` di-*fit* pada data latih saja, lalu diterapkan ke data uji. **Urutan benar, tidak ada leakage.**
7. **Feature Selection** — dua pendekatan, masing-masing memilih **10 fitur**:
   - **SelectKBest** dengan `f_classif` (ANOVA F-value) — metode filter, menilai fitur secara individual.
   - **RFE** (Recursive Feature Elimination) — metode wrapper, menilai fitur berdasarkan kontribusinya pada model. Dijalankan dengan tiga estimator berbeda: Logistic Regression, Random Forest, SVM linear.
8. **Modeling** — 6 kombinasi: 3 algoritma × 2 metode seleksi.
9. **Evaluasi** — Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC.

**Kenapa dua metode seleksi fitur?** Dataset ini punya 30 fitur yang saling berkorelasi tinggi — `radius_mean`, `perimeter_mean`, dan `area_mean` pada dasarnya mengukur hal yang sama. SelectKBest menilai fitur satu per satu sehingga cenderung memilih fitur yang redundan; RFE menilai fitur dalam konteks model sehingga bisa membuang duplikat. Membandingkan keduanya menunjukkan metode mana yang lebih tahan terhadap multikolinearitas.

**Fitur terpilih:**

| Metode | Fitur |
|---|---|
| **SelectKBest** | `radius_mean`, `perimeter_mean`, `area_mean`, `concavity_mean`, `concave points_mean`, `radius_worst`, `perimeter_worst`, `area_worst`, `concavity_worst`, `concave points_worst` |
| **RFE + LR** | `concave points_mean`, `radius_se`, `area_se`, `compactness_se`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `concavity_worst`, `concave points_worst` |
| **RFE + RF** | `radius_mean`, `perimeter_mean`, `area_mean`, `concavity_mean`, `concave points_mean`, `radius_worst`, `texture_worst`, `perimeter_worst`, `area_worst`, `concave points_worst` |
| **RFE + SVM** | `concavity_mean`, `concave points_mean`, `radius_se`, `texture_se`, `area_se`, `compactness_se`, `fractal_dimension_se`, `texture_worst`, `area_worst`, `concavity_worst` |

Perhatikan bahwa SelectKBest dan RFE+RF memilih fitur yang hampir identik (10 vs 10, hanya beda satu), sementara RFE+LR dan RFE+SVM memilih campuran `_mean` dan `_se` yang berbeda. Ini menunjukkan **pilihan fitur bergantung pada model yang dipakai** — bukan sifat absolut dari data.

---

## 📊 Temuan Utama (EDA)

### 1. Fitur bentuk dan ukuran sel mendominasi

Korelasi terhadap target menunjukkan fitur berbasis **bentuk dan ukuran** sebagai yang paling informatif:

| Fitur | Korelasi \|r\| terhadap diagnosis |
|---|---:|
| `concave points_worst` | Tertinggi |
| `perimeter_worst` | Sangat tinggi |
| `radius_worst` | Sangat tinggi |
| `area_worst` | Sangat tinggi |
| `concave points_mean` | Tinggi |

Sebaliknya, beberapa fitur punya korelasi sangat rendah (< 0,1) — misalnya `fractal_dimension_mean` dan `texture_se` — dan kemungkinan besar tidak informatif untuk model.

**Interpretasi medis:** tumor malignant cenderung memiliki batas yang **tidak teratur** (nilai `concave points` tinggi) dan **ukuran lebih besar** (`perimeter`, `radius`, `area`). Ini sejalan dengan pemahaman klinis bahwa sel ganas tumbuh tidak terkendali dan kehilangan bentuk normalnya.

### 2. Multikolinearitas ekstrem antar fitur

`radius_mean`, `perimeter_mean`, dan `area_mean` saling berkorelasi sangat tinggi (ketiganya mengukur ukuran yang sama dengan satuan berbeda). Ini alasan utama seleksi fitur diperlukan — tanpa itu, model harus menangani 30 fitur yang sebagian besar redundan.

### 3. Distribusi target cukup seimbang

Dengan 62,7% benign dan 37,3% malignant, dataset ini **tidak** memerlukan SMOTE atau penyeimbangan kelas. Baseline acak berada di 62,7% akurasi, dan target 95% berarti model harus jauh melampaui itu.

| Data Loading | Distribusi Target |
|---|---|
| ![Data Loading](image/Data%20Loading.jpg) | ![Distribusi Target](image/Dsitribusi%20fitur%20taget.jpg) |

| Heatmap Korelasi |
|---|
| ![Heatmap](image/heatmap_korelasi.png) |

---

## 🤖 Hasil Model

### Perbandingan 6 Kombinasi Model

| Rank | Model | Accuracy | Precision | Recall | F1 Score |
|---:|---|---:|---:|---:|---:|
| 🥇 | **RFE + Logistic Regression** | 0,9737 | 0,9756 | **0,9524** | **0,9639** |
| 🥈 | SelectKBest + SVM | 0,9737 | 1,0000 | 0,9286 | 0,9630 |
| 🥈 | RFE + Random Forest | 0,9737 | 1,0000 | 0,9286 | 0,9630 |
| 4 | SelectKBest + Logistic Regression | 0,9561 | 0,9744 | 0,9048 | 0,9383 |
| 5 | SelectKBest + Random Forest | 0,9561 | 1,0000 | 0,8810 | 0,9367 |
| 6 | RFE + SVM | 0,9474 | 0,9737 | 0,8810 | 0,9250 |

**Nilai absolut tidak terlalu penting di sini — yang penting adalah selisihnya.** Empat dari enam model punya akurasi ≥ 95,6%. Semua model "terlihat bagus". Pembedanya ada di confusion matrix.

### Confusion Matrix — di sinilah keputusan sesungguhnya

Test set: 114 sampel (72 benign, 42 malignant).

| Model | TN | FP | **FN** | TP |
|---|---:|---:|---:|---:|
| **RFE + Logistic Regression** | 71 | 1 | **2** | 40 |
| SelectKBest + SVM | 72 | 0 | 3 | 39 |
| RFE + Random Forest | 72 | 0 | 3 | 39 |
| SelectKBest + Logistic Regression | 71 | 1 | 4 | 38 |
| RFE + SVM | 71 | 1 | **5** | 37 |
| SelectKBest + Random Forest | 72 | 0 | **5** | 37 |

**Cara membaca untuk konteks medis:**

- **RFE + Logistic Regression** melewatkan **2 dari 42** kasus kanker — terbaik.
- **RFE + SVM** dan **SelectKBest + Random Forest** melewatkan **5 dari 42** — lebih dari dua kali lipat.
- Selisih akurasi antara model terbaik dan terlemah hanya **2,6 poin persentase** (97,37% vs 94,74%), tapi selisih kasus kanker yang terlewat adalah **2 vs 5** — lebih dari dua kali lipat.

Inilah alasan **recall dan confusion matrix harus dibaca lebih dulu daripada akurasi** dalam konteks diagnosis.

### ROC-AUC & PR-AUC

| Model | ROC-AUC | PR-AUC |
|---|---:|---:|
| **SelectKBest + SVM** | **0,9987** | **0,9978** |
| RFE + Logistic Regression | 0,9974 | 0,9957 |
| SelectKBest + Logistic Regression | 0,9974 | 0,9956 |
| RFE + Random Forest | 0,9945 | 0,9916 |
| RFE + SVM | 0,9944 | 0,9913 |
| SelectKBest + Random Forest | 0,9894 | 0,9848 |

> **Ada dua "model terbaik" di sini, dan itu wajar.** SelectKBest + SVM unggul di ROC-AUC/PR-AUC (kemampuan *meranking* probabilitas), sedangkan RFE + Logistic Regression unggul di recall pada threshold 0,5 (kemampuan *memutuskan*). ROC-AUC mengukur kualitas pemisahan di semua threshold; recall mengukur hasil pada satu threshold tertentu. Untuk sistem yang memberi keputusan ya/tidak, **recall lebih relevan**. Untuk sistem yang mengurutkan pasien berdasarkan prioritas, **ROC-AUC lebih relevan**.

| Evaluasi Performa Model |
|---|
| ![Evaluasi](image/Evaluasi%20performa%20model.jpg) |

---

## ⚠️ Keterbatasan & Asumsi

### 1. Test set hanya 114 sampel — selisih antar model belum tentu nyata

Ini keterbatasan paling penting. Dengan hanya **42 kasus malignant** di test set, selisih 2 vs 5 false negative berada dalam rentang yang bisa muncul dari **kebetulan pemilihan split** saja.

Interval kepercayaan 95% (Wilson) untuk recall:

| Model | Recall | 95% CI |
|---|---:|---|
| RFE + Logistic Regression | 0,9524 (40/42) | **0,842 – 0,987** |
| SelectKBest + SVM | 0,9286 (39/42) | 0,810 – 0,975 |
| RFE + SVM | 0,8810 (37/42) | 0,750 – 0,948 |

**Interval-interval ini saling tumpang tindih secara lebar.** Artinya: peringkat "RFE + LR terbaik" **belum terbukti signifikan secara statistik**. Untuk mengklaim itu, perlu **cross-validation berulang** (mis. 10-fold × 10 repeat) dengan interval kepercayaan pada selisih performa.

### 2. Single train-test split

Seluruh hasil berasal dari satu split (`random_state=42`). Tidak ada cross-validation, sehingga tidak ada estimasi variabilitas performa. Ini standar minimum untuk klaim "model terbaik" yang serius.

### 3. Tidak ada validasi eksternal

Dataset berasal dari satu institusi dengan satu protokol pengukuran. Model belum diuji pada data dari rumah sakit, alat, atau populasi pasien lain. Performa pada domain baru **belum diketahui** dan bisa turun signifikan.

### 4. Hyperparameter belum dioptimalkan

`GridSearchCV` diimpor tetapi tidak dipakai. Semua model memakai hyperparameter default (Logistic Regression tanpa tuning `C`, Random Forest default, SVM `kernel='linear'` default). Perbandingan ini adil sebagai baseline, tapi belum tentu mencerminkan potensi maksimal tiap algoritma.

### 5. Seleksi fitur dilakukan pada data latih — sudah benar, tapi perlu dicatat

`SelectKBest` dan `RFE` di-*fit* pada `X_train_scaled` saja, lalu di-*transform* ke test. Ini **sudah benar** dan menghindari leakage. Namun seleksi fitur dilakukan **di luar** cross-validation, sehingga pada evaluasi CV nanti proses ini harus dimasukkan ke dalam pipeline.

### 6. Klaim "siap dipakai di rumah sakit" terlalu kuat

Dengan 569 pasien, satu split, dan tanpa validasi eksternal, model ini adalah **proof of concept**, bukan alat diagnosis klinis. Diperlukan validasi prospektif, uji pada populasi beragam, dan persetujuan regulator sebelum dipakai pada pasien nyata.

### 7. Korelasi bukan kausalitas

`concave points_worst` berkorelasi dengan diagnosis, tetapi ini bukan berarti ia "menyebabkan" kanker. Fitur-fitur ini adalah **penanda morfologi** yang diamati, bukan mekanisme biologis.

---

## 💡 Rekomendasi

### Untuk Pengembangan Model (prioritas utama)

1. **Jalankan cross-validation berulang** (Stratified 10-fold × 10 repeat) dan laporkan mean ± std untuk setiap model. Ini akan menunjukkan apakah keunggulan RFE + LR benar-benar nyata atau hanya efek split.
2. **Masukkan seleksi fitur ke dalam `Pipeline`** agar prosesnya ikut tervalidasi di setiap fold — bukan dilakukan sekali di luar CV.
3. **Lakukan hyperparameter tuning** dengan `GridSearchCV`, terutama `C` untuk Logistic Regression dan `n_estimators`/`max_depth` untuk Random Forest.
4. **Optimalkan ambang keputusan (threshold).** Threshold default 0,5 tidak optimal untuk prioritas recall. Kurva precision-recall bisa menunjukkan titik di mana recall naik ke 0,98 dengan penurunan precision yang masih dapat diterima.
5. **Uji pada dataset eksternal** — minimal dataset Wisconsin versi lain atau data dari institusi berbeda — untuk mengukur seberapa jauh performa turun di luar domain latih.

### Untuk Implementasi Klinis (dengan syarat)

1. **Posisikan model sebagai alat bantu skrining, bukan pengganti diagnosis.** Keputusan akhir tetap pada dokter spesialis patologi.
2. **Fokus pada recall, bukan akurasi.** Jika model harus memilih, lebih baik menandai lebih banyak kasus sebagai "perlu diperiksa" daripada melewatkan satu kanker.
3. **Sertakan probabilitas, bukan hanya label.** Dokter perlu tahu apakah model 51% atau 99% yakin — keduanya akan dilabeli "malignant" pada threshold 0,5, tapi implikasinya sangat berbeda.
4. **Bangun mekanisme audit.** Catat setiap prediksi dan hasil konfirmasi patologi untuk memantau performa model setelah dipakai (monitoring drift).

### KPI yang Disarankan

| KPI | Baseline (test set) | Target |
|---|---|---|
| Recall kelas malignant | 0,9524 (2 FN / 42) | ≥ 0,98 (≤ 1 FN) |
| False negative per 1.000 pasien | ~18 (ekstrapolasi) | < 10 |
| Precision kelas malignant | 0,9756 | ≥ 0,95 (jaga FP tetap rendah) |
| ROC-AUC (CV mean) | Belum diukur | ≥ 0,99 stabil antar fold |
| Waktu analisis per kasus | Manual (menit) | < 1 detik |

---

## 🚀 Cara Menjalankan

### 1. Prasyarat

- Python **3.10+**
- Akun Kaggle + `kaggle.json` (untuk mengunduh dataset)

### 2. Install dependensi

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

### 3. Siapkan dataset

```bash
kaggle datasets download -d uciml/breast-cancer-wisconsin-data
unzip breast-cancer-wisconsin-data.zip
```

### 4. Jalankan

**Opsi A — Notebook (disarankan):**

```bash
jupyter notebook project1_pretictive_analytic.ipynb
```

**Opsi B — Script Python:**

```bash
python project1_pretictive_analytic.py
```

> **Catatan:** script `.py` hasil ekspor Colab masih memuat baris `files.upload()` dan perintah `!kaggle` / `!unzip` yang hanya jalan di Colab. Untuk eksekusi lokal, unduh dataset secara manual dan sesuaikan path `read_csv`.

---

## 🗂️ Struktur Proyek

```
breast-cancer-prediction/
│
├── project1_pretictive_analytic.ipynb   # Notebook utama
├── project1_pretictive_analytic.py      # Ekspor Colab
├── README.md
│
└── image/
    ├── Data Loading.jpg
    ├── Dsitribusi fitur taget.jpg
    ├── heatmap_korelasi.png
    └── Evaluasi performa model.jpg
```

---

## 🛠️ Tech Stack

| Kategori | Tools |
|---|---|
| Bahasa | Python 3.10+ |
| Analisis data | pandas, NumPy |
| Machine learning | scikit-learn (Logistic Regression, Random Forest, SVM, SelectKBest, RFE) |
| Visualisasi | Matplotlib, Seaborn |
| Environment | Jupyter / Google Colab |

---

## 📚 Referensi

[1] Sung, H., Ferlay, J., Siegel, R. L., Laversanne, M., Soerjomataram, I., Jemal, A., & Bray, F. (2021). Global cancer statistics 2020: GLOBOCAN estimates of incidence and mortality worldwide for 36 cancers in 185 countries. *CA: A Cancer Journal for Clinicians*, 71(3), 209–249. https://doi.org/10.3322/caac.21660

[2] Kumar, R., & Rani, R. (2019). Feature Selection Method To Improve The Accuracy of Classification. *International Journal of Innovative Technology and Exploring Engineering (IJITEE)*, 8(6), 342–345. https://www.ijitee.org/wp-content/uploads/papers/v8i6/F3421048619.pdf

[3] Kumar, A., & Sharma, R. (2020). Recursive Feature Elimination with Feature Ranking-Based Feature Selection. *International Journal of Innovative Research in Technology (IJIRT)*, 6(2), 223–226. https://ijirt.org/publishedpaper/IJIRT162237_PAPER.pdf

[4] AbdelGawad, A., & Ratner, D. (2007). Adaptive Optimization of Hyperparameters in L2-Regularised Logistic Regression. *CS229 Project Report, Stanford University*. https://cs229.stanford.edu/proj2007/AbdelGawadRatner-AdaptiveHyperparameterOptimization.pdf

[5] Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5–32. https://doi.org/10.1023/A:1010933404324

[6] Panda, B. S. (n.d.). Support Vector Machines. *Indian Institute of Technology Delhi*. https://web.iitd.ac.in/~bspanda/SVM.pdf

[7] Dauphin, G. (2019). Confusion Matrix. *Laboratoire d'Informatique de Paris Nord*. https://www-l2ti.univ-paris13.fr/~dauphin/Confusion_matrix.pdf

---

## 👤 Author

**Zulfahmi M. Ardianto**
Laskar ID: A548YBM523
