read me
mlp_quantum_optimization/
│
├── data/
│   └── mnist/            # MNIST veri seti veya işlenmiş versiyonlar
│
├── models/
│   ├── classical_mlp.py  # Classical MLP model kodu
│   ├── qmlp_vqc.py       # Quantum MLP (QMLP) ve VQC kodu
│
├── training/
│   ├── train_classical.py  # Classical model için eğitim scripti
│   ├── train_quantum.py    # Quantum model için eğitim scripti
│
├── utils/
│   ├── dataset.py        # MNIST yükleme, dimension reduction fonksiyonları
│   ├── plots.py          # Loss, accuracy grafikleri çizimi
│   ├── encoding.py       # Quantum data encoding fonksiyonları
│
├── experiments/
│   ├── benchmarks.ipynb  # Karşılaştırma deneyleri, analizler
│   └── results/          # Eğitim logları, grafikleri, sonuç tabloları
│
├── requirements.txt      # Kütüphane listesi
├── README.md             # Proje açıklaması
├── .gitignore            # Cache, temp dosyaları dışarıda tutmak için
│
└── main.py               # Ana kontrol scripti (train/test seçimi, config yükleme vs.)