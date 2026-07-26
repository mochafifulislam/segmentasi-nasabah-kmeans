import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def run_clustering_pipeline(input_path: str, output_path: str):
    print("=== [1/5] Loading Dataset ===")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File tidak ditemukan pada path: {input_path}")
        
    df = pd.read_csv(input_path)
    df_copy = df.copy()

    print("=== [2/5] Data Preprocessing & Normalization ===")
    # Identifikasi kolom kategorikal
    categorical_columns = df_copy.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Label Encoding untuk variabel kategorikal
    encoder = LabelEncoder()
    for col in categorical_columns:
        df_copy[col] = encoder.fit_transform(df_copy[col].astype(str))
    print(f"Kolom kategorikal dikonversi: {len(categorical_columns)} kolom")

    # Handling Missing Values
    df_copy.fillna(df_copy.mean(), inplace=True)

    # Scaling MinMaxScaler
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_copy), columns=df_copy.columns)

    print("=== [3/5] Determining Optimal Cluster (Elbow Method) ===")
    wcss = []
    K_range = range(1, 11)
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(df_scaled)
        wcss.append(kmeans.inertia_)

    # Hitung jumlah cluster optimal secara otomatis berdasarkan penurunan diferensial
    diff = np.diff(wcss)
    diff_r = diff[1:] / diff[:-1]
    k_optimal = np.argmin(diff_r) + 2
    print(f"Cluster optimal terdeteksi: {k_optimal}")

    print("=== [4/5] Executing K-Means Clustering ===")
    kmeans_model = KMeans(n_clusters=k_optimal, random_state=42, n_init=10)
    cluster_labels = kmeans_model.fit_predict(df_scaled)
    
    # Menambahkan label cluster ke dataset asli
    df["Kelompok"] = cluster_labels

    # Simpan Hasil Clustering
    df.to_csv(output_path, index=False)
    print(f"Hasil clustering berhasil disimpan ke: {output_path}")

    print("=== [5/5] Visualizing Clusters with PCA ===")
    pca = PCA(n_components=2)
    components = pca.fit_transform(df_scaled)
    
    plt.figure(figsize=(9, 6))
    for cluster in range(k_optimal):
        plt.scatter(
            components[cluster_labels == cluster, 0],
            components[cluster_labels == cluster, 1],
            label=f'Cluster {cluster}',
            alpha=0.7
        )
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Visualisasi PCA Segmentasi Nasabah BSI")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    INPUT_FILE = "data/data_nasabah.csv"
    OUTPUT_FILE = "data/clustered_data_nasabah.csv"
    
    # Jalankan pipeline
    run_clustering_pipeline(INPUT_FILE, OUTPUT_FILE)
