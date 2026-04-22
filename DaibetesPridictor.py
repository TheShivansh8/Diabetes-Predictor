from flask import Flask, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    target_column = request.form['target']
    data = pd.read_csv(file)

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lin_reg = LinearRegression().fit(X_train, y_train)
    ridge = Ridge(alpha=1.0).fit(X_train, y_train)
    lasso = Lasso(alpha=0.1).fit(X_train, y_train)

    results = {
        "Linear MSE": mean_squared_error(y_test, lin_reg.predict(X_test)),
        "Linear R2": r2_score(y_test, lin_reg.predict(X_test)),
        "Ridge MSE": mean_squared_error(y_test, ridge.predict(X_test)),
        "Ridge R2": r2_score(y_test, ridge.predict(X_test)),
        "Lasso MSE": mean_squared_error(y_test, lasso.predict(X_test)),
        "Lasso R2": r2_score(y_test, lasso.predict(X_test)),
    }

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
