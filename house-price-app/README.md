# House Price Estimator

A Flask website wrapping a scikit-learn Random Forest model trained
on the Ames Housing dataset (1,460 sales, R² ≈ 0.886, RMSE ≈ $29.6k).

## ⚠️ Important — how to open this site

**Do NOT double-click `templates/index.html` to open it in your browser.**
This is not a static website — it's a Flask app, and the page needs a
running Python server behind it to fill in the neighborhood list, the
model stats, and the price calculation. If you open the HTML file
directly, you'll see broken placeholder text like `{{ n.name }}` on
screen and the "Estimate price" button won't do anything — that's the
tell that the server isn't running.

## Run it correctly

1. Open a terminal (Command Prompt / PowerShell on Windows, Terminal on Mac)
2. Navigate into the unzipped folder:
   ```bash
   cd path/to/house-price-app
   ```
3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Start the server:
   ```bash
   python app.py
   ```
5. You should see output like:
   ```
   * Running on http://127.0.0.1:5000
   ```
6. Now open **http://127.0.0.1:5000** in your browser — not the HTML file.

Leave the terminal window open the whole time you're using the site;
closing it stops the server.

### If it still looks broken

- See `{{ ... }}` text on the page → you opened the HTML file directly,
  not the localhost URL. Go back to step 6.
- Slider stuck at one number → same cause, the JS file isn't loading.
- "Could not reach the server" when estimating → the terminal running
  `python app.py` was closed, or `pip install` didn't finish — rerun it.

## Retrain the model

The trained model is already included (`model/house_price_model.pkl`), but
if you want to retrain (e.g. with a different dataset or feature set):

```bash
python train.py
```

This re-fits Linear Regression and Random Forest, prints RMSE/R² for both,
and saves whichever one scores higher to `model/`.

## Project structure

```
app.py              Flask server (routes: / and /predict)
train.py            Training script — compares Linear Regression vs Random Forest
model/
  house_price_model.pkl   Saved sklearn Pipeline (preprocessing + model)
  meta.json                Feature ranges, neighborhood list, model metrics
templates/index.html      Single-page form + result panel
static/style.css          Design (colors, layout, icons)
static/script.js          Fetches /predict and renders the result
```

## Deploying

Works as-is on Render, Railway, or PythonAnywhere — just point them at
`app.py` with the `requirements.txt`. For Render, add a `gunicorn` line to
requirements and set the start command to `gunicorn app:app`.
