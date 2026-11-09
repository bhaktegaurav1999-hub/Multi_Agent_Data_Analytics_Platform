
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


DATA_FILE = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "sales_data.csv"
)


def load_sales_data():
    df = pd.read_csv(DATA_FILE)
    df["order_date"] = pd.to_datetime(df["order_date"])
    return df


def prepare_data():
    df = load_sales_data()

    features = [
        "quantity",
        "unit_price",
        "discount",
    ]

    X = df[features]
    y = df["sales"]

    return X, y


def train_sales_model():
    X, y = prepare_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    return model, mae, r2


def get_feature_importance():
    model, _, _ = train_sales_model()

    features = [
        "quantity",
        "unit_price",
        "discount",
    ]

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_,
    })

    return importance.sort_values(
        "importance",
        ascending=False,
    )


def predict_sales(quantity, unit_price, discount):
    model, _, _ = train_sales_model()

    input_data = pd.DataFrame({
        "quantity": [quantity],
        "unit_price": [unit_price],
        "discount": [discount],
    })

    prediction = model.predict(input_data)[0]

    return round(prediction, 2)


# Run this section only when this file is executed directly.
# It will not run when another module imports this file.
if __name__ == "__main__":

    model, mae, r2 = train_sales_model()

    print("\nSales Prediction Model")
    print("----------------------")
    print(f"Mean Absolute Error: {mae:.2f}")
    print(f"R2 Score: {r2:.4f}")

    print("\nFeature Importance")
    print("------------------")
    print(get_feature_importance())

    print("\nNew Order Sales Prediction")
    print("---------------------------")

    prediction = predict_sales(
        quantity=5,
        unit_price=1000,
        discount=0.10,
    )

    print(f"Predicted Sales: {prediction:.2f}")
