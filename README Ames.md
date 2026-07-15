#  Ames Home Value Estimator

An end-to-end machine learning web application that predicts residential
sale prices from property characteristics, trained on the **Ames Housing
dataset** (1,460 real home sales in Ames, Iowa). The project covers the
full ML pipeline — data cleaning, preprocessing, model comparison, and
deployment — wrapped in a live, interactive web interface.

---

##  Overview

Given a handful of property details (square footage, quality rating,
garage size, neighborhood, etc.), the app returns an estimated sale
price along with a confidence range, computed by a trained regression
model served through a Flask REST API.

The dataset itself has **79 raw features**, but exposing all of them in
a form isn't usable. Instead, the 10 most predictive features were
identified via feature-importance analysis and used to build a clean,
practical interface without a meaningful loss in accuracy.

---

##  Features

- Clean, responsive single-page form for entering property details
- Real-time price prediction via a REST API (no page reload)
- Plain-language explanations under each input field
- Model performance (R², RMSE) surfaced directly in the UI
- Full training pipeline included and reproducible from raw data

---

##  Model & Performance

Two regression models were trained and compared on an 80/20 train-test
split:

| Model              | RMSE       | R²     |
|---------------------|-----------|--------|
| Linear Regression    | $36,056  | 0.831  |
| **Random Forest**    | **$29,617** | **0.886** |

**Random Forest** was selected as the production model based on its
lower error and higher explained variance.

**Features used (top 10 by importance):**
`OverallQual`, `GrLivArea`, `TotalBsmtSF`, `GarageCars`, `YearBuilt`,
`FullBath`, `LotArea`, `TotRmsAbvGrd`, `YearRemodAdd`, `Neighborhood`

---

##  Tech Stack

| Layer            | Tools |
|-------------------|-------|
| Language           | Python |
| ML / Data          | scikit-learn, pandas, NumPy |
| Backend            | Flask (REST API) |
| Model persistence  | joblib |
| Frontend           | HTML, CSS, JavaScript (vanilla, no framework) |
| Templating         | Jinja2 |

---

##  Project Structure

```
house-price-app/
├── app.py                    # Flask server — routes: / and /predict
├── train.py                  # Training script (Linear Regression vs Random Forest)
├── requirements.txt
├── model/
│   ├── house_price_model.pkl # Saved sklearn pipeline (preprocessing + model)
│   └── meta.json             # Feature ranges, neighborhood list, model metrics
├── templates/
│   └── index.html            # Main page (form + result panel)
└── static/
    ├── style.css
    └── script.js
```

---

##  How It Works

1. **`train.py`** loads the raw dataset, builds a preprocessing pipeline
   (`SimpleImputer` → `StandardScaler`/`OneHotEncoder` via
   `ColumnTransformer`), trains both models, evaluates them, and saves
   the better-performing pipeline to `model/house_price_model.pkl`.
2. **`app.py`** loads that saved pipeline at startup and exposes two
   routes:
   - `GET /` — renders the form, populated with live feature ranges and
     neighborhood options from `model/meta.json`
   - `POST /predict` — accepts JSON input, runs it through the pipeline,
     and returns a predicted price plus a confidence range (± RMSE)
3. The frontend (`script.js`) sends form data to `/predict` via
   `fetch()` and updates the price on screen without a page reload.

---

##  Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/ames-home-value-estimator.git
cd ames-home-value-estimator
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
python app.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

> **Note:** This is a Flask app, not a static site — `templates/index.html`
> must be served by `app.py`. Opening the HTML file directly in a browser
> won't work, since the page relies on server-rendered data and a live
> API endpoint.

### (Optional) Retrain the model
```bash
python train.py
```
This re-fits both models on the raw dataset, prints RMSE/R² for each,
and overwrites `model/house_price_model.pkl` with whichever performs
better.

---

##  Dataset

[Ames Housing Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)
— 1,460 residential property sales in Ames, Iowa, with 79 explanatory
variables covering nearly every aspect of the home (size, quality,
condition, location, year built, and more).

---

##  Future Improvements

- Add more features to the model with a progressive/advanced input mode
- Model explainability (e.g. SHAP values) to show *why* a price was predicted
- Deploy to a public URL (Render/Railway) for live access
- Add input validation and error states for edge cases

---

## 👤 Author

**Niranjan S**
[LinkedIn](https://linkedin.com/in/niranjan-s) · niranjan.syamsj@gmail.com
