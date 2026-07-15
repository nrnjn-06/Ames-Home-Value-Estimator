import json
import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL = joblib.load("model/house_price_model.pkl")
with open("model/meta.json") as f:
    META = json.load(f)

FEATURES = META["features"]

# Full neighborhood names, since the raw codes (e.g. "NAmes") aren't
# self-explanatory. Values sent to the model still use the short code.
NEIGHBORHOOD_NAMES = {
    "Blmngtn": "Bloomington Heights",
    "Blueste": "Bluestem",
    "BrDale": "Briardale",
    "BrkSide": "Brookside",
    "ClearCr": "Clear Creek",
    "CollgCr": "College Creek",
    "Crawfor": "Crawford",
    "Edwards": "Edwards",
    "Gilbert": "Gilbert",
    "IDOTRR": "Iowa DOT & Rail Road",
    "MeadowV": "Meadow Village",
    "Mitchel": "Mitchell",
    "NAmes": "North Ames",
    "NPkVill": "Northpark Villa",
    "NWAmes": "Northwest Ames",
    "NoRidge": "Northridge",
    "NridgHt": "Northridge Heights",
    "OldTown": "Old Town",
    "SWISU": "South & West of Iowa State University",
    "Sawyer": "Sawyer",
    "SawyerW": "Sawyer West",
    "Somerst": "Somerset",
    "StoneBr": "Stone Brook",
    "Timber": "Timberland",
    "Veenker": "Veenker",
}


@app.route("/")
def index():
    neighborhoods = [
        {"code": code, "name": NEIGHBORHOOD_NAMES.get(code, code)}
        for code in META["neighborhoods"]
    ]
    neighborhoods.sort(key=lambda n: n["name"])

    return render_template(
        "index.html",
        neighborhoods=neighborhoods,
        ranges=META["ranges"],
        model_name=META["model_name"],
        r2=round(META["r2"], 3),
        rmse=round(META["rmse"]),
    )



@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    try:
        row = {
            "OverallQual": int(data["OverallQual"]),
            "GrLivArea": float(data["GrLivArea"]),
            "TotalBsmtSF": float(data["TotalBsmtSF"]),
            "GarageCars": int(data["GarageCars"]),
            "YearBuilt": int(data["YearBuilt"]),
            "FullBath": int(data["FullBath"]),
            "LotArea": float(data["LotArea"]),
            "TotRmsAbvGrd": int(data["TotRmsAbvGrd"]),
            "YearRemodAdd": int(data["YearRemodAdd"]),
            "Neighborhood": data["Neighborhood"],
        }
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    X = pd.DataFrame([row])[FEATURES]
    pred = float(MODEL.predict(X)[0])
    rmse = META["rmse"]

    return jsonify({
        "prediction": round(pred),
        "low": round(pred - rmse),
        "high": round(pred + rmse),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
