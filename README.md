# IF3270 Tugas Besar II - Machine Learning

Repositori ini berisi implementasi dari model **Convolutional Neural Network (CNN)**, **Simple Recurrent Neural Network (Simple RNN)**, dan **Long-Short Term Memory (LSTM)** untuk tugas klasifikasi menggunakan dataset **CIFAR-10** dan **NusaX-Sentiment**. Semua model diuji dengan berbagai variasi arsitektur dan hyperparameter, serta dilengkapi dengan modul forward propagation dari awal (from scratch).

## Struktur Direktori

```
src/
│
├── convolutional-neural-network/
│   ├── histories/
│   ├── models/
│   ├── results/
│   ├── experiment_results.pkl
│   ├── layers.py
│   ├── main.ipynb
│   └── model.py
│
├── recurrent-neural-network/
│   ├── results/
│   ├── layers.py
│   ├── main.ipynb
│   ├── model.py
│   └── utils.py
│
├── long-short-term-memory-network/
│   ├── results/
│   ├── layers.py
│   ├── main.ipynb
│   ├── model_comp.weights.h5
│   ├── model.py
│   └── utils.py
│
└── K03_G19_Laporan-Tubes-2.pdf
```

## Cara Setup dan Menjalankan Program

1. **Clone repository dan masuk ke direktori project**:
   ```bash
   git clone https://github.com/namapengguna/nama-repo.git
   cd nama-repo
   ```

2. **Buat environment Python dan aktifkan**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Linux/macOS
   .venv\Scripts\activate         # Windows
   ```

3. **Install dependency yang dibutuhkan**:
   ```bash
   pip install requirements
   ```

4. **Jalankan notebook untuk masing-masing model**:
   - CNN: `src/convolutional-neural-network/main.ipynb`
   - Simple RNN: `src/recurrent-neural-network/main.ipynb`
   - LSTM: `src/long-short-term-memory-network/main.ipynb`

## Pembagian Tugas

| NIM        | Nama Lengkap             | Tugas                            |
|------------|--------------------------|----------------------------------|
| 13522130   | Justin Aditya Putra P.   | Implementasi & pengujian LSTM   |
| 13522155   | Axel Santadi Warih       | Implementasi & pengujian RNN    |
| 13522163   | Atqiya Haydar Luqman     | Implementasi & pengujian CNN    |

## Referensi

- CIFAR-10: https://www.tensorflow.org/datasets/catalog/cifar10  
- NusaX Sentiment: https://github.com/IndoNLP/nusax  
- Keras dan TensorFlow documentation  
- Materi kuliah IF3270 Machine Learning
