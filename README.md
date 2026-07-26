# 🏦 Segmentasi Nasabah BSI KCP Batamindo Menggunakan K-Means Clustering

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458)

Project ini bertujuan untuk melakukan **segmentasi pelanggan/nasabah** pada Bank Syariah Indonesia (BSI) KCP Batamindo menggunakan algoritma unsupervised learning **K-Means Clustering**. Dengan segmentasi ini, pihak manajemen dan *marketing* perbankan dapat menganalisis karakteristik portofolio nasabah, meningkatkan efektivitas strategi pemasaran, serta menawarkan produk finansial secara tepat sasaran.

---

## 📌 Pendahuluan & Latar Belakang

Di industri perbankan modern, pendekatan *one-size-fits-all* untuk strategi penawaran produk sudah tidak efektif. Setiap nasabah memiliki profil demografi, pola transaksi, tingkat keaktifan aplikasi digital (seperti *byond*), serta kisaran rata-rata saldo (*tier average balance*) yang berbeda.

Melalui pendekatan berbasis Machine Learning, proyek ini mengelompokkan data nasabah ke dalam kelompok-kelompok homogen (klaster) untuk membantu identifikasi kriteria nasabah potensial, pengguna aktif payroll, maupun nasabah yang memerlukan penetrasi produk baru.

---

## 📊 Fitur & Data yang Digunakan

Dataset berisi variabel demografi, kepemilikan produk, indikator keaktifan, dan informasi wilayah perbankan:

* **Informasi Demografi & Identitas**: `gender`, `alamat`, `institution_nm`
* **Keaktifan Produk & Layanan**:
  * `flag_payroll` (Status penerima payroll)
  * `flag_byond` (Penggunaan super-app BSI BYOND)
  * `flag_debitcard`, `flag_debitcard_expired`, `debitcard_expired_dt`
  * `flag_cilem`, `flag_mabrur`, `flag_usak_bsim`
* **Keuangan**: `tier_avg_balance` (Tingkat/kisaran rata-rata saldo)
* **Informasi Area & Cabang**: `cabangbsi`, `areabsi`, `regionbsi`

---

## ⚙️ Metodologi & Pipeline Kode

Pipeline pemrosesan data dan algoritma clustering mengikuti tahapan berikut:

### 1. Data Preprocessing & Scaling
* **Categorical Encoding**: Konversi seluruh kolom ber-tipe kategorikal (`object`/`category`) menjadi bentuk numerik menggunakan `LabelEncoder`.
* **Handling Missing Values**: Pengisian nilai hilang (*NaN*) menggunakan nilai rerata (*mean imputation*).
* **Feature Scaling**: Normalisasi rentang fitur numerik menggunakan **MinMaxScaler** agar perbedaan skala numerik tidak mendominasi perhitungan jarak Euclidean pada K-Means:

  $$X_{scaled} = \frac{X - X_{min}}{X_{max} - X_{min}}$$

### 2. Penetapan Jumlah Cluster Optimal ($K$)
Menggunakan **Elbow Method** berdasarkan nilai *Within-Cluster Sum of Squares* (WCSS):

$$WCSS = \sum_{i=1}^{K} \sum_{x \in C_i} \|x - \mu_i\|^2$$

Evaluasi penurunan varians otomatis menentukan titik optimal pada **$K = 8$ cluster**.

### 3. Clustering & Reduksi Dimensi
* Diterapkan algoritma **K-Means** dengan $K=8$.
* Visualisasi klaster dalam ruang 2D menggunakan **Principal Component Analysis (PCA)** untuk menyederhanakan visualisasi multivariat.

---

## 💡 Interpretasi Bisnis & Profiling Cluster

Berdasarkan hasil pengelompokan ($K=8$), berikut adalah contoh strategi bisnis dan rekomendasi aksi yang dapat diterapkan per segmen:

| Cluster | Profil Utama Nasabah | Karakteristik Produk & Saldo | Rekomendasi Pemasaran / Action Plan |
| :---: | :--- | :--- | :--- |
| **Cluster 0** | Nasabah Payroll Aktif Digital | Saldo sedang–tinggi, `flag_payroll` = 1, `flag_byond` = 1 | Penawaran pembiayaan konsumtif (Mitraguna/Griya) via aplikasi BYOND. |
| **Cluster 1** | Nasabah Low-Tier Non-Digital | Saldo relatif rendah, belum aktivasi `flag_byond` | Eductational campaign aktivasi aplikasi BYOND & insentif promo transaksi. |
| **Cluster 2** | Nasabah Priority / High Balance | `tier_avg_balance` tinggi, aktif kartu debit | Layanan *wealth management*, penawaran Tabungan Mabrur / Haji & Umrah. |
| **Cluster 3** | Institutional Payroll Client | Terikat institusi tertentu, `flag_payroll` = 1 | Retensi hubungan B2B institusi & penawaran program pembiayaan khusus karyawan. |
| **Cluster 4-7**| Segmen Transaksional Khusus | Bervariasi berdasarkan status kartu debit & wilayah | Re-aktivasi kartu debit kadaluarsa & penawaran fitur perbankan syariah lainnya. |

---

## 🚀 Cara Menjalankan Proyek

### 1. Clone Repositori
```bash
git clone [https://github.com/mochafifulislam/segmentasi-nasabah-bsi-kmeans.git](https://github.com/mochafifulislam/segmentasi-nasabah-bsi-kmeans.git)
cd segmentasi-nasabah-bsi-kmeans
