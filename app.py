from flask import Flask, render_template, request, redirect, send_file
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

app = Flask(__name__)

# Tải các mô hình đã huấn luyện
model_mlp = joblib.load('models\\neural_network_(mlpregressor).pkl')  # MLP model
model_ridge = joblib.load('models\\ridge_regression.pkl')     # Ridge Regression model
model_lr = joblib.load('models\\linear_regression.pkl')  # Linear Regression model
model_stacking = joblib.load('models\\stacking_model.joblib')  # Stacking model

# Tải lại X_test và y_test từ file CSV
X_test_processed = pd.read_csv('data/X_test.csv')
y_test = pd.read_csv('data/y_test.csv')

# Định nghĩa tên các cột đặc trưng mà mô hình của bạn đã huấn luyện
column_names = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# File để lưu lịch sử sử dụng
history_file = 'data/history.csv'

# Tính toán độ lệch chuẩn của sai số trên tập kiểm tra
def calculate_standard_error(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.sqrt(mse)

# Route chính để hiển thị trang web
@app.route('/')
def index():
    history = pd.read_csv(history_file) if os.path.exists(history_file) else pd.DataFrame()
    return render_template('index.html', history=history.to_dict(orient='records'))

@app.route('/predict', methods=['POST'])
def predict():
    # Lấy dữ liệu từ form của người dùng
    model_choice = request.form['model_choice']
    age = request.form['age']
    bmi = request.form['bmi']
    children = request.form['children']
    smoker = request.form['smoker']
    sex = request.form['sex']
    region = request.form['region']

    # Tạo DataFrame đầu vào cho mô hình
    input_features = pd.DataFrame([[float(age), sex, float(bmi), int(children), smoker, region]], columns=column_names)

    # Chọn mô hình dựa trên lựa chọn của người dùng
    if model_choice == 'mlp':
        model = model_mlp
    elif model_choice == 'ridge':
        model = model_ridge
    elif model_choice == 'stacking':
        model = model_stacking
    else:
        model = model_lr

    # Dự đoán chi phí bảo hiểm
    predicted_charge = model.predict(input_features)[0]

    # Tính toán sai số chuẩn
    std_error = calculate_standard_error(model, X_test_processed, y_test)
    confidence_interval = 1.96 * std_error
    lower_bound = predicted_charge - confidence_interval
    upper_bound = predicted_charge + confidence_interval

    # Lưu lịch sử sử dụng
    new_record = {
        'model': model_choice,
        'age': age,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'sex': sex,
        'region': region,
        'predicted_charge': predicted_charge,
        'confidence_interval': f'(${lower_bound:.2f}, ${upper_bound:.2f})'
    }

    # Kiểm tra xem file lịch sử có tồn tại không, nếu không thì tạo mới
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
    else:
        history = pd.DataFrame(columns=new_record.keys())

    # Thay thế append bằng pd.concat
    history = pd.concat([history, pd.DataFrame([new_record])], ignore_index=True)

    # Ghi lại file CSV
    history.to_csv(history_file, index=False)

    return render_template('index.html',
                           prediction_text=f'Estimated Insurance Charge: ${predicted_charge:.2f}',
                           confidence_text=f'95% Confidence Interval: (${lower_bound:.2f}, ${upper_bound:.2f})',
                           age=age, bmi=bmi, children=children, smoker=smoker, sex=sex, region=region,
                           model_choice=model_choice,
                           history=history.to_dict(orient='records'))

# Route để tải xuống lịch sử
@app.route('/download_history')
def download_history():
    return send_file(history_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
