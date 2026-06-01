# 🚀 KarzMitra – AI-Powered Loan Risk Predictor

Predict loan risk instantly using Machine Learning and receive clear, human-readable explanations powered by Generative AI.

---

## 📌 Overview

**KarzMitra** is a full-stack AI web application designed to assess loan risk by combining traditional Machine Learning with modern Large Language Models (LLMs).

The system not only predicts whether an applicant is **High Risk** or **Low Risk**, but also explains the reasoning behind the decision in plain English—making the process transparent and easier to understand for lenders and users.

This project was developed as part of the **Google Solution Challenge 2026**.

---

## ✨ Key Features

* 🤖 **ML-Based Risk Prediction**
  Classifies applicants into *High Risk* or *Low Risk* using a trained model

* 🧠 **AI-Powered Explanation**
  Generates detailed, human-friendly insights using an LLM

* 🌐 **Interactive Web Interface**
  Clean and responsive UI built with Flask and templating

* 📊 **End-to-End Workflow**
  Includes data preprocessing, model training, and inference pipelines

---

## 🛠️ Tech Stack

| Layer            | Technologies Used                  |
| ---------------- | ---------------------------------- |
| Backend          | Python, Flask                      |
| Machine Learning | scikit-learn, pandas, numpy, scipy |
| Generative AI    | LLM API (via `llm_engine.py`)      |
| Frontend         | HTML5, CSS3, Jinja2                |
| Data & Analysis  | Jupyter, matplotlib                |
| Deployment       | Gunicorn, Render                   |
| Configuration    | python-dotenv                      |

---

## 📁 Project Structure

```
Solution_challenge/
│
├── app.py                  # Main Flask application
├── requirements.txt        # Dependencies
│
├── src/
│   ├── preprocess.py       # Data preprocessing logic
│   ├── predictor.py        # ML prediction module
│   └── llm_engine.py       # LLM explanation engine
│
├── models/                 # Trained model files
├── data/                   # Dataset files
├── notebooks/              # EDA & training notebooks
├── templates/              # HTML templates
├── static/                 # CSS & assets
```

---

## ⚙️ How It Works

1. User enters financial details via the web interface
2. Input data is preprocessed and transformed
3. ML model predicts loan risk (High / Low)
4. Result is passed to the LLM engine
5. LLM generates a detailed explanation
6. Final output is displayed to the user in real-time

---

## 💻 Running the Project Locally

### Prerequisites

* Python 3.10+
* API key for an LLM provider

### Steps

```bash
# Clone the repository
git clone https://github.com/Raj-UtsaV/Solution_challenge.git
cd Solution_challenge

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
# Create a .env file and add:
API_KEY=your_api_key_here
BASE_URL=your_api_endpoint
LLM_MODEL=your_model_name

# Run the application
python app.py
```

Open in browser:
👉 http://localhost:5000

---

## 🌐 Live Demo

👉 https://solution-challenge-eg1f.onrender.com/

> Note: The app is hosted on a free tier and may take a few seconds to load initially.

---

## 👨‍💻 Contributions

This project was developed collaboratively as part of the Google Solution Challenge 2026.

### Contributors

* Utsav Raj
* Aditya Bansal
* Shivay Saurya

### My Contributions

* Contributed to development and integration of core modules
* Assisted in improving model workflow and application structure
* Worked on enhancing usability and overall system functionality

---

## 📢 Disclaimer

The AI-generated explanations depend on the availability of a valid API key.
The ML prediction module works independently even without the LLM integration.

---

## ⭐ Future Improvements

* Model performance optimization
* Enhanced UI/UX
* Support for multiple financial datasets
* More explainability features

---

## ❤️ Acknowledgment

Built with the aim of making credit risk assessment more transparent, explainable, and intelligent.

---
