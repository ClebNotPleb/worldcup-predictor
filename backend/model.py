"""
model.py - World Cup Match Prediction Model
Trains a Random Forest classifier on historical World Cup data
and saves it for use by the FastAPI backend.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

FIFA_RANKINGS = {
    "Argentina": 1875, "France": 1877, "Spain": 1876, "England": 1826,
    "Portugal": 1764, "Brazil": 1761, "Netherlands": 1758, "Morocco": 1756,
    "Belgium": 1735, "Germany": 1730, "Croatia": 1717, "Colombia": 1693,
    "Senegal": 1689, "Mexico": 1681, "United States": 1673, "Uruguay": 1673,
    "Japan": 1660, "Switzerland": 1649, "Denmark": 1621, "Norway": 1610,
    "Australia": 1605, "South Korea": 1598, "Ecuador": 1590, "Canada": 1585,
    "Turkey": 1580, "Serbia": 1570, "Ukraine": 1560, "Austria": 1555,
    "Scotland": 1550, "Sweden": 1545, "Czechia": 1540, "Poland": 1535,
    "Bosnia and Herzegovina": 1525, "Ghana": 1520, "Tunisia": 1515,
    "Algeria": 1510, "Egypt": 1505, "Nigeria": 1500, "Ivory Coast": 1495,
    "Cameroon": 1490, "Mali": 1485, "Cape Verde": 1480, "Burkina Faso": 1475,
    "South Africa": 1470, "Zambia": 1460, "Iran": 1520, "Uzbekistan": 1450,
    "Jordan": 1440, "Saudi Arabia": 1430, "Iraq": 1420, "Qatar": 1400,
    "New Zealand": 1510, "Paraguay": 1540, "Venezuela": 1490, "Bolivia": 1450,
    "Peru": 1530, "Chile": 1545, "Costa Rica": 1480, "Panama": 1470,
    "Jamaica": 1440, "Honduras": 1430, "El Salvador": 1420, "Haiti": 1410,
    "Curacao": 1390, "Guatemala": 1380, "New Caledonia": 1350,
    "DR Congo": 1465, "Benin": 1440, "Angola": 1430, "Tanzania": 1410,
}

WORLD_CUP_TITLES = {
    "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3,
    "France": 2, "Uruguay": 2, "England": 1, "Spain": 1,
}

def get_titles(team):
    return WORLD_CUP_TITLES.get(team, 0)

def get_ranking(team):
    return FIFA_RANKINGS.get(team, 1400)

def generate_synthetic_data(n_samples=3000):
    np.random.seed(42)
    teams = list(FIFA_RANKINGS.keys())
    records = []

    for _ in range(n_samples):
        home = np.random.choice(teams)
        away = np.random.choice(teams)
        if home == away:
            continue

        home_rank = get_ranking(home)
        away_rank = get_ranking(away)
        rank_diff = home_rank - away_rank

        base_prob_home = 0.45 + (rank_diff / 10000)
        base_prob_draw = 0.25
        base_prob_away = max(0.05, 1 - base_prob_home - base_prob_draw)

        total = base_prob_home + base_prob_draw + base_prob_away
        probs = [base_prob_home/total, base_prob_draw/total, base_prob_away/total]

        result = np.random.choice(["Home Win", "Draw", "Away Win"], p=probs)

        records.append({
            "home_ranking": home_rank,
            "away_ranking": away_rank,
            "ranking_diff": rank_diff,
            "home_titles": get_titles(home),
            "away_titles": get_titles(away),
            "result": result,
        })

    return pd.DataFrame(records)

def train_model(data_path=None):
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        print("🔧 Generating training data...")
        df = generate_synthetic_data()

    print(f"✅ Dataset size: {len(df)} matches")

    features = ["home_ranking", "away_ranking", "ranking_diff", "home_titles", "away_titles"]
    X = df[features]
    y = df["result"]

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=8)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {acc:.2%}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/match_predictor.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("💾 Model saved!")

def predict_match(home_team, away_team):
    model = joblib.load("models/match_predictor.pkl")
    le = joblib.load("models/label_encoder.pkl")

    features = pd.DataFrame([{
        "home_ranking": get_ranking(home_team),
        "away_ranking": get_ranking(away_team),
        "ranking_diff": get_ranking(home_team) - get_ranking(away_team),
        "home_titles": get_titles(home_team),
        "away_titles": get_titles(away_team),
    }])

    proba = model.predict_proba(features)[0]
    classes = le.classes_

    result = {cls: round(float(prob)*100, 1) for cls, prob in zip(classes, proba)}
    predicted = classes[proba.argmax()]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_ranking": get_ranking(home_team),
        "away_ranking": get_ranking(away_team),
        "probabilities": result,
        "predicted_winner": predicted,
        "home_titles": get_titles(home_team),
        "away_titles": get_titles(away_team),
    }

if __name__ == "__main__":
    train_model()