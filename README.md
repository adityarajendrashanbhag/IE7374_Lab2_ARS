# 🌸 Iris Flower Classifier — IE7374 Lab 2

A full-stack machine learning application that classifies Iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) using two models: a **TensorFlow Neural Network** and a **Scikit-learn Random Forest**. The app is served via a **Flask** REST API with a cyberpunk-themed web UI, and is fully containerized with **Docker Compose**.

---

## 📁 Repository Structure

```
IE7374_Lab2_ARS/
├── docker-compose.yml          # Multi-service orchestration (train → serve)
├── dockerfile                   # Single-container image for the Flask API
├── requirements.txt             # Python dependencies
├── README.md
└── src/
    ├── main.py                  # Flask application (API + web routes)
    ├── model_training.py        # TensorFlow model training script
    ├── train_rf.py              # RandomForest model training script
    ├── my_model.keras           # Trained TensorFlow model artifact
    ├── tf_scaler.joblib         # StandardScaler for TF model inputs
    ├── rf_model.joblib          # Trained RandomForest model artifact
    ├── scaler.joblib            # StandardScaler for RF model inputs
    ├── templates/
    │   ├── predict.html         # Web UI for TensorFlow predictions
    │   └── predict_rf.html      # Web UI for RandomForest predictions
    └── statics/
        ├── setosa.jpeg          # Iris Setosa reference image
        ├── versicolor.jpeg      # Iris Versicolor reference image
        └── virginica.jpeg       # Iris Virginica reference image
```

---

## ✅ Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| pip | Latest |
| Docker | 20.10+ (for container runs) |
| Docker Compose | v2+ (for multi-service runs) |

---

## 🚀 Getting Started

### Option 1 — Run Locally (Windows)

**1. Create and activate a virtual environment:**

CMD:
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

PowerShell:
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

> If you already have the provided venv (`docker-lab2-venv`), activate it instead:
> `docker-lab2-venv\Scripts\activate.bat`

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. Train the models:**

```bash
cd src
python model_training.py
python train_rf.py
```

This produces four artifact files in the `src/` directory:
- `my_model.keras` — TensorFlow neural network
- `tf_scaler.joblib` — Scaler used during TF training
- `rf_model.joblib` — RandomForest classifier
- `scaler.joblib` — Scaler used during RF training

**4. Start the Flask API:**

```bash
python main.py
```

**5. Open in your browser:**

| Page | URL |
|---|---|
| Home | http://127.0.0.1:4000/ |
| TensorFlow Predictor | http://127.0.0.1:4000/predict |
| RandomForest Predictor | http://127.0.0.1:4000/predict_rf |

---

### Option 2 — Docker (Single Container)

**1. Build the image:**

```bash
docker build -t iris-api .
```

**2. Run the container:**

```bash
docker run --rm -p 4000:4000 iris-api
```

The app will be available at **http://localhost:4000**.

---

### Option 3 — Docker Compose (Multi-Service)

Docker Compose runs two services in sequence:

1. **model-training** — Trains both the TF and RF models and saves artifacts to a shared volume.
2. **serving** — Copies trained artifacts from the shared volume and launches the Flask API.

**1. Build and start:**

```bash
docker-compose up --build
```

**2. Access the app:**

The app will be available at **http://localhost:80**.

**3. Stop and clean up:**

```bash
# Stop containers
docker-compose down

# Stop and remove trained model volume (forces retraining)
docker-compose down -v
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check / welcome message |
| `GET` | `/predict` | Serves the TensorFlow prediction web UI |
| `POST` | `/predict` | Returns TF prediction (form-encoded) |
| `GET` | `/predict_rf` | Serves the RandomForest prediction web UI |
| `POST` | `/predict_rf` | Returns RF prediction (form-encoded or JSON) |

### Example POST Request

```bash
curl -X POST http://localhost:4000/predict \
  -d "sepal_length=5.1&sepal_width=3.5&petal_length=1.4&petal_width=0.2"
```

**Response:**

```json
{
  "predicted_class": "Setosa"
}
```

---

## 🧠 Models

### TensorFlow Neural Network (`model_training.py`)
- Architecture: 2-layer Dense network (8 neurons → 3 softmax outputs)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Epochs: 50
- Input preprocessing: StandardScaler (saved as `tf_scaler.joblib`)

### Random Forest Classifier (`train_rf.py`)
- Estimators: 100
- Input preprocessing: StandardScaler (saved as `scaler.joblib`)
- Trained on the same 80/20 split with `random_state=42`

Both models are trained on the classic [Iris dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-dataset) (150 samples, 4 features, 3 classes).

---

## ⚠️ Troubleshooting

| Issue | Solution |
|---|---|
| `/predict` or `/predict_rf` returns 503 | Model files are missing. Run the training scripts first or rebuild Docker containers. |
| `deactivate` not recognized on Windows CMD | Use `venv\Scripts\deactivate.bat` or just close the terminal. |
| Port 4000 already in use | Stop other services on that port, or change the port in `src/main.py`. |
| Docker Compose serves on port 80 but `docker run` on 4000 | This is by design — `docker-compose.yml` maps 80→4000, while running locally uses 4000 directly. |

---

## 📝 Notes

- The Flask app listens on **0.0.0.0:4000** by default (configurable in `src/main.py`).
- If any model artifact file is missing, the corresponding endpoint gracefully returns a **503** error instead of crashing.
- The web UIs display a reference image of the predicted Iris species after classification.
