# 🚰 Water Quality Prediction – AICTE Virtual Internship (June 2025)

This project predicts multiple water quality parameters using machine learning techniques, specifically a `MultiOutputRegressor` wrapped around a `RandomForestRegressor`. It was developed as part of the AICTE Virtual Internship conducted by Edunet Foundation and sponsored by Shell in June 2025.

Live Demo: https://water-quality-predictionaicte-shnhd9rzxyjkgakhyzgyah.streamlit.app/
---

## 🌍 Overview

Access to clean water is a critical global challenge. Accurate prediction of water quality metrics enables early pollution detection and informed decision-making for environmental monitoring.

### In this project, we:

- Collected and cleaned real-world water quality datasets
- Applied supervised machine learning for **multi-target regression**
- Built a pipeline using `MultiOutputRegressor(RandomForestRegressor)`
- Visualized the data distribution and missing values
- Evaluated the model using multiple regression metrics

---

## 🧪 Predicted Parameters

The model predicts the following water quality parameters:

- NH4 (Ammonium)
- BOD5 (BSK5 – Biochemical Oxygen Demand)
- Suspended Colloids
- O₂ (Dissolved Oxygen)
- NO₃ (Nitrate)
- NO₂ (Nitrite)
- SO₄ (Sulfate)
- PO₄ (Phosphate)
- CL (Chloride)

---

## 🔧 Technologies Used

- **Python 3.12** – Core programming language  
- **Jupyter Notebook** – Interactive development and experimentation  
- **Libraries:**
  - `pandas`, `numpy` – Data preprocessing and manipulation
  - `scikit-learn` – Machine learning modeling and evaluation
  - `matplotlib`, `seaborn` – Visualization

---

## 📈 Model Evaluation

The model was evaluated using the following metrics:

- **R² Score** – Measures the goodness-of-fit for each target
- **Mean Squared Error (MSE)** – Evaluates prediction accuracy

Performance was found to be satisfactory across all predicted parameters, with robust results from the Random Forest model.

---

## 📁 Repository Structure

Water-Quality-Prediction_AICTE/
├── data/ # Raw or processed datasets (if shared)
├── images/ # Visualizations, heatmaps, etc.
├── notebooks/ # Jupyter Notebooks for EDA and modeling
├── models/ # Saved model files (if any)
├── README.md # Project overview
├── requirements.txt # Python dependencies
└── water_quality_model.ipynb # Main notebook

---

## 🚀 Getting Started

### 1. Clone the repository
git clone https://github.com/AmitAK1/Water-Quality-Prediction_AICTE.git

cd Water-Quality-Prediction_AICTE

2. Install dependencies
Make sure you have Python 3.12+ installed. Then run:

pip install -r requirements.txt

3. Run the notebook
Open the main notebook in Jupyter:

jupyter notebook water_quality_model.ipynb
🎓 Internship Details
Internship Type: AICTE Virtual Internship

Organizer: Edunet Foundation

Sponsor: Shell

Duration: June 2025 (1 month)

Domain: Machine Learning in Environmental Monitoring

📌 Future Scope
Deploy as a Streamlit web app for public use

Add missing value imputation strategies

Explore other ML algorithms like Gradient Boosting, SVR

Integrate with sensor data for real-time predictions

👤 Author
Amit Kamble
🌐 https://github.com/AmitAK1/Water-Quality-Prediction_AICTE
