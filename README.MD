# 🏙️ NYC Room Type Prediction

A machine learning web application that predicts the **room type of an Airbnb listing in New York City** based on property, location, pricing, booking, review, and host-related features.

The application is built entirely with **Streamlit**, allowing users to enter property information and receive a real-time prediction along with class probabilities and model confidence.

---

## 🚀 Live Demo

🔗 **Streamlit App:**
https://nyc-air-bnb-ml-based.streamlit.app/

---

## 📌 Project Overview

Airbnb listings contain various attributes that can help machine learning models identify the type of accommodation being offered.

This project uses a trained machine learning classification pipeline to predict one of the following room types:

* Entire home/apt
* Private room
* Shared room

The application provides an interactive interface where users can enter listing details and instantly receive a prediction.

---

## 🎯 Problem Statement

Given information about an Airbnb listing, predict its **room type** using machine learning.

The model uses features related to:

* Geographic location
* Neighbourhood
* Price
* Minimum nights
* Number of reviews
* Reviews per month
* Host listing count
* Availability

This can be treated as a **multiclass classification problem**.

---

## 🧠 Machine Learning Workflow

The project follows an end-to-end machine learning workflow:

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Feature Engineering
     ↓
Feature Selection
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Pipeline Creation
     ↓
Model Serialization
     ↓
Streamlit Deployment
```

---

## 📊 Features Used

The trained model uses the following features:

| Feature                          | Description                            |
| -------------------------------- | -------------------------------------- |
| `latitude`                       | Latitude coordinate of the listing     |
| `longitude`                      | Longitude coordinate of the listing    |
| `price`                          | Price per night                        |
| `minimum_nights`                 | Minimum number of nights required      |
| `number_of_reviews`              | Total number of reviews                |
| `reviews_per_month`              | Average reviews received per month     |
| `calculated_host_listings_count` | Number of listings managed by the host |
| `availability_365`               | Number of available days in a year     |
| `neighbourhood_group`            | NYC borough/neighbourhood group        |
| `neighbourhood`                  | Specific neighbourhood                 |

---

## 🏗️ Application Architecture

This project intentionally uses a **Streamlit-only architecture**.

```text
                    USER
                      │
                      ▼
              ┌───────────────┐
              │   Streamlit   │
              │      UI       │
              └───────┬───────┘
                      │
                      ▼
             Input Validation
                  Pydantic
                      │
                      ▼
              Pandas DataFrame
                      │
                      ▼
          Model_Pipeline.pkl
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    model.predict()       predict_proba()
          │                       │
          └───────────┬───────────┘
                      ▼
             Prediction Result
                      │
                      ▼
              Streamlit Dashboard
```

### Why Streamlit Only?

The application does not require a separate backend API.

Instead of:

```text
Streamlit → FastAPI → ML Model
```

the application directly performs:

```text
Streamlit → ML Model
```

This makes the application simpler to deploy and maintain for this use case.

---

## 🖥️ Application Features

### 1. Property Input

Users can provide:

* Neighbourhood group
* Neighbourhood
* Latitude
* Longitude
* Price
* Minimum nights
* Availability
* Number of reviews
* Reviews per month
* Host listing count

### 2. Input Validation

The application uses **Pydantic** to validate user inputs before sending them to the model.

Examples:

```text
Latitude:
-90 ≤ latitude ≤ 90

Longitude:
-180 ≤ longitude ≤ 180

Price:
price > 0

Minimum nights:
1 ≤ minimum_nights ≤ 365

Availability:
0 ≤ availability_365 ≤ 365
```

### 3. Room Type Prediction

The trained pipeline generates the predicted room type.

Example:

```text
Predicted Room Type
Private room
```

### 4. Model Confidence

The application displays the highest predicted class probability.

Example:

```text
Model Confidence
87.42%
```

### 5. Prediction Probability

The application displays probabilities for each room type.

Example:

| Room Type       | Probability |
| --------------- | ----------: |
| Entire home/apt |      72.41% |
| Private room    |      26.83% |
| Shared room     |       0.76% |

### 6. Visualization

Prediction probabilities are displayed using a Streamlit bar chart.

### 7. Property Summary

After prediction, the application displays a summary of the submitted property information.

---

## 🛠️ Technologies Used

### Programming Language

* Python

### Machine Learning

* Scikit-learn
* Pandas
* NumPy
* Joblib

### Application

* Streamlit

### Validation

* Pydantic

### Deployment

* Streamlit Community Cloud

---

## 📁 Project Structure

```text
NYC-Room-Type-Prediction/
│
├── app.py
│
├── Model_Pipeline.pkl
│
├── requirements.txt
│
├── .gitignore
│
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Move into the project directory:

```bash
cd NYC-Room-Type-Prediction
```

---

### 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

### 4. Run the Application

```powershell
streamlit run app.py
```

The application will open in your browser.

---

## 📦 Requirements

The project uses:

```text
streamlit
pandas
pydantic
scikit-learn
joblib
```

---

## 🤖 Model Loading

The trained machine learning pipeline is loaded using Joblib:

```python
@st.cache_resource
def load_model():
    return joblib.load("Model_Pipeline.pkl")
```

Using `st.cache_resource` prevents Streamlit from unnecessarily reloading the trained model every time the application reruns.

---

## 🔮 Prediction Process

When the user clicks **Predict Room Type**, the application:

```text
1. Collects user inputs
        ↓
2. Validates the inputs
        ↓
3. Creates a Pandas DataFrame
        ↓
4. Passes the DataFrame to the trained pipeline
        ↓
5. Generates the room type prediction
        ↓
6. Calculates class probabilities
        ↓
7. Displays the prediction
        ↓
8. Displays probability visualization
        ↓
9. Displays property summary
```

---

## 📈 Prediction Code

The core prediction process is:

```python
row = pd.DataFrame(
    [features.model_dump()],
    columns=COLUMNS
)

prediction = model.predict(row)

probabilities = model.predict_proba(row)

predicted_room = prediction[0]
```

The trained pipeline handles the required preprocessing and prediction steps.

---

## 🌐 Deployment

The application can be deployed using **Streamlit Community Cloud**.

General deployment process:

```text
GitHub Repository
        ↓
Streamlit Community Cloud
        ↓
Connect Repository
        ↓
Select app.py
        ↓
Deploy
```

Make sure `requirements.txt` and `Model_Pipeline.pkl` are included in the repository.

---

## 🔒 Input Validation

The application validates user input before prediction.

For example:

```python
latitude: float = Field(..., ge=-90, le=90)

longitude: float = Field(..., ge=-180, le=180)

price: float = Field(..., gt=0)

minimum_nights: int = Field(..., ge=1, le=365)
```

This helps prevent invalid values from being passed directly to the model.

---

## ⚠️ Important Notes

The model's predictions depend on the quality and distribution of the data used during training.

A prediction probability should be interpreted as the model's estimated class probability, **not as a guarantee that the prediction is correct**.

The application is intended for machine learning demonstration and portfolio purposes.

---

## 🔮 Future Improvements

Potential improvements include:

* Add interactive NYC map visualization
* Add model performance metrics
* Add confusion matrix visualization
* Add feature importance analysis
* Add SHAP-based model explainability
* Add prediction history
* Add batch prediction using CSV upload
* Add downloadable prediction reports
* Improve UI responsiveness
* Add automated model retraining
* Add experiment tracking
* Add monitoring for deployed predictions

---

## 📚 Skills Demonstrated

This project demonstrates experience with:

* Machine Learning
* Multiclass Classification
* Data Preprocessing
* Feature Engineering
* Model Pipelines
* Model Serialization
* Pydantic Validation
* Pandas
* Scikit-learn
* Streamlit
* Data Visualization
* Model Deployment
* Git & GitHub

---

## 👩‍💻 Author

**Manya Rajput**

B.Tech Computer Science Engineering

GitHub:
https://github.com/Manya-Rajput-2007

LinkedIn:
https://linkedin.com/in/manya-rajput-2a281b33b

---

## ⭐ If You Find This Project Useful

Feel free to explore the repository, experiment with the model, and improve the application.

If you find the project useful, consider giving the repository a ⭐.


