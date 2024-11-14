from flask import Flask, request, send_file, jsonify, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__, static_url_path='/static')

# Load dataset (change this to the path of your CSV file)
df = pd.read_csv('dataset.csv')

# Load the prediction model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('prediction.html')

@app.route('/generate-chart', methods=['POST'])
def generate_chart():
    data = request.get_json()
    visualization_type = data['visualizationType']
    graph_type = data['graphType']

    # Chart data based on the selected visualization type
    if visualization_type == 'gender_distribution':
        chart_data = df['Gender'].value_counts()
        title = 'Gender Distribution'
    elif visualization_type == 'income_distribution':
        chart_data = df['Total_income'].value_counts(bins=10)
        title = 'Income Distribution'
    elif visualization_type == 'employment_status':
        chart_data = df['Unemployed'].value_counts()
        title = 'Employment Status Distribution'
    elif visualization_type == 'age_distribution':
        chart_data = pd.cut(df['Age'], bins=5).value_counts()
        title = 'Age Distribution'

    # Create the plot
    plt.figure(figsize=(6.8, 6))
    if graph_type == 'bar':
        bars = chart_data.plot(kind='bar')
        y_label = 'Number of Entries'
        for bar in bars.patches:
            bars.annotate(f'{int(bar.get_height())}', 
                          (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                          ha='center', va='bottom')
    elif graph_type == 'pie':
        labels = [f'{index} ({count})' for index, count in zip(chart_data.index, chart_data.values)]
        chart_data.plot(kind='pie', labels=labels, startangle=90, legend=False)
        y_label = ''
        plt.axis('equal')

    plt.title(title)
    plt.ylabel(y_label)

    # Save the plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close()  # Close the plot to free memory

    return send_file(img, mimetype='image/png')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)
    final_features = scaler.transform(final_features)    
    prediction = model.predict(final_features)
    y_probabilities = model.predict_proba(final_features)
    probability = round(y_probabilities[0, 1], 3)  # Probability of positive class

    output = round(prediction[0], 2)
    
    if output == 0:
        prediction_text = 'Prediction: Likely Negative Outcome with Probability {}'.format(probability)
    else:
        prediction_text = 'Prediction: Likely Positive Outcome with Probability {}'.format(probability)
    
    return render_template('prediction.html', prediction_text=prediction_text)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

if __name__ == "__main__":
    app.run(debug=True)
