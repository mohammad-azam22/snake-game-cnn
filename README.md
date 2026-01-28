# Snake Game Controlled by Hand-Drawn Gestures (CNN) 🐍🖊️

An end-to-end project that integrates **deep learning**, **API deployment**, and **interactive UI** to control a browser-based Snake game using hand-drawn letter gestures (`L`, `R`, `U`, `D`).  
This system uses a custom **Convolutional Neural Network (CNN)** for image classification and maps model predictions to game controls.


## 🧠 Overview

This repository demonstrates a complete machine learning pipeline — from dataset creation to model deployment and real-time user interaction.

Instead of using normal keyboard controls, the player draws a direction (`L`, `R`, `U`, `D`) on a canvas, and the CNN classifies the gesture and controls the snake accordingly.

![UI image](./images/Home%20image.png)

---

## 🧩 Features

- 🎨 **Custom dataset** of 4,000 hand-drawn gesture images (1,000 per class)
- 🤖 Multiple CNN model experiments with **MLflow tracking**
- 📈 Best model selected based on accuracy and loss
- 🐍 A responsive **Snake game** playable on desktop and mobile
- 🚀 Deployed as a **Dockerized Flask web service**
- 🖥️ Real-time gesture classification via REST API

---

## 📁 Repository Structure

├── `app.py` # Flask API server<br>
├── `inference.py` # Model inference logic<br>
├── `model.py` # CNN architecture definition<br>
├── `preprocessing.py` # Image preprocessing functions<br>
├── `requirements.txt` # Python dependencies<br>
├── `train.py` # Model training and MLflow<br>
├── `Dockerfile` # Docker config for deployment<br>
├── `index.html` # Frontend UI<br>
├── `/data/train/` # Dataset folders<br>
├── `/models/` # Saved trained model artifacts<br>
├── `/images/` # Images used in UI<br>
├── `LICENSE`<br>
└── `README.md`


---

## 📌 Highlights

### ✔ Dataset

- Hand-drawn gesture images representing `L`, `R`, `U`, `D`
- Balanced dataset: 1000 images per class
- Images are grayscale and resized to `64×64`

### ✔ Model Experiments

Trained and compared:

| Model Type | Notes | Train Accuracy | Train Loss | Validation Accuracy | Validation Loss |
|------------|-------|----------------|------------|---------------------|-----------------|
| Multi-Layer Perceptron | Baseline | 1.0000  | 4.5727e-04 | 0.9725 | 0.1883 |
| Simple CNN | Small CNN architecture | 1.0000 | 0.0010 | 0.9885 | 0.0491 |
| Complex CNN | Deeper model | 1.0000 | 8.0177e-04 | 0.9962 | 0.0110 |
| CNN + Augmentation + Resampling + Dropout | **Best performer** | 0.9959 | 0.0127 | 0.9987 | 1.9727e-04 |

Best model achieved:

- Train Accuracy: **0.9959**
- Validation Accuracy: **0.9987**
- Low validation loss with strong generalization

Experiment tracking and comparison were done using **MLflow**.

---

## 🚀 Deployment

The best CNN model was saved and deployed behind a REST API endpoint (`/predict`) using **Flask**. The container was deployed on **Render.com** via a Docker image built from this repository.

---

## 🕹️ Game Integration

The Snake game UI (HTML/CSS/JS) includes:

- Canvas for drawing gestures
- Mobile-friendly touch controls
- Real-time classification requests
- Movement control based on model predictions

The game sends drawn images to the API, receives predictions, and updates the snake’s direction accordingly.

---

## 💻 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/mohammad-azam22/snake-game-cnn.git
cd snake-game-cnn
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Start Flask server
```bash
waitress-serve --host=0.0.0.0 --port=9696 app:app
```
Server will run at http://localhost:9696

### 4. Open Game UI

Open index.html in a browser and draw gestures on the canvas to control the snake.

## 📌 Predict API Example

Endpoint: `/predict`<br>
Method: `POST`<br>
Body: `Canvas drawing`

Response:
`{
    prediction: confidence
}`
<br>
example: 
`{
  "D": "98.3",
  "L": "0.04",
  "R": "1.0",
  "U": "0.03"
}`

## 🧠 Tech Stack
| Component | Tech |
|-----------|------|
| Model Training | TensorFlow / CNN |
|Experiment Tracking | MLflow |
| API Framework | Flask |
| Containerization | Docker |
| Deployment | Render |
| Frontend | HTML, CSS, JavaScript |
| Game Logic | JavaScript |

## ⭐ Credits

This project was built from scratch starting from data collection to deployment, combining classical game logic with modern deep learning for interactive controls. It demonstrates practical ML engineering, model serving, and frontend integration.

## 📄 License

This repository is licensed under the MIT License. See the license tab for more details.
