
# 🏥 Insurance Cost Prediction System (ML + Gradio)

A Machine Learning project that predicts medical insurance costs based on user inputs like age, BMI, smoking status, and more.  
Built using Python, Scikit-learn, and Gradio for interactive web UI.

---

## 🚀 Run the Project

### 1. Clone repo
```bash
git clone https://github.com/your-username/insurance-ml-project.git
cd insurance-ml-project
````

---

### 2. Create virtual environment

```bash
python -m venv venv
```

---

### 3. Activate virtual environment

Windows:

```bash
.venv\Scripts\activate
```

Mac/Linux:

```bash
source venv/bin/activate
```

---

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 5. Run ML training (optional)

```bash
python model_training.py
```

---

### 6. Run Gradio App

```bash
python app.py
```

Then open:

```
http://127.0.0.1:7860
```

---

## 📊 Dataset Features

* age → Age of person
* sex → Gender
* bmi → Body Mass Index
* children → Number of children
* smoker → Smoking status
* region → Residential region
* bmi_category → Engineered feature

---

## 🧠 Machine Learning Pipeline

### Data Processing

* Missing value handling
* Outlier detection (IQR method)
* Feature engineering (BMI category)

### Encoding & Scaling

* OneHotEncoding for categorical data
* StandardScaler for numerical data

### Models Used

* Linear Regression
* Random Forest
* Gradient Boosting
* Voting Regressor
* Stacking Regressor

---

## 🏆 Best Model Performance

* R² Score: ~0.84
* RMSE: ~4100
* MAE: ~2200

Best Model:

* Gradient Boosting Regressor

---

## 🌐 Gradio App

Input:

* Age
* Sex
* BMI
* Children
* Smoker
* Region

Output:

* Predicted Insurance Cost

---

## 💾 Model Saving

Model saved as:

```
insurance_model.pkl
```

---

## 📁 Project Structure

```
insurance_model/
│
├── model_training.py
├── app.py
├── insurance_model.pkl
├── insurance.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Gradio
* Pickle


