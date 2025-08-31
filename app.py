
from flask import Flask, render_template, request, flash, redirect, url_for
import pickle
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
app.secret_key = 'diabetes_predictor_secret_key'

# Map old module path to new one for compatibility
try:
    sys.modules['sklearn.ensemble.forest'] = sys.modules['sklearn.ensemble._forest']
    sys.modules['sklearn.tree.tree'] = sys.modules['sklearn.tree']
except:
    pass

# Load the trained model
try:
    filename = 'D:\\projects\\sid\\diabetes_prediction_rfc.pkl'
    with open(filename, 'rb') as f:
        classifier = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    classifier = None

@app.route('/')
def home():
    return render_template('D:\\projects\\sid\\siddd.venv\\diabites\\index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':
            # Get form data with validation
            preg = float(request.form.get('pregnancies', 0))
            glucose = float(request.form.get('glucose', 0))
            bp = float(request.form.get('bloodpressure', 0))
            st = float(request.form.get('skinthickness', 0))
            insulin = float(request.form.get('insulin', 0))
            bmi = float(request.form.get('bmi', 0))
            dpf = float(request.form.get('dpf', 0))
            age = float(request.form.get('age', 0))
            
            # Basic validation
            if any(val < 0 for val in [preg, glucose, bp, st, insulin, bmi, dpf, age]):
                flash('Please enter valid positive values for all fields.', 'error')
                return redirect(url_for('home'))
            
            if age < 1 or age > 120:
                flash('Please enter a valid age between 1 and 120 years.', 'error')
                return redirect(url_for('home'))
            
            if bmi < 10 or bmi > 70:
                flash('Please enter a valid BMI between 10 and 70.', 'error')
                return redirect(url_for('home'))
            
            # Check if model is loaded
            if classifier is None:
                flash('Model not available. Please try again later.', 'error')
                return redirect(url_for('home'))
            
            # Prepare data for prediction
            data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
            
            # Make prediction
            my_prediction = classifier.predict(data)[0]  # Get the first (and only) prediction
            
            return render_template('D:\\projects\\sid\\siddd.venv\\diabites\\result.html', prediction=int(my_prediction))
            
    except ValueError as e:
        flash('Please enter valid numerical values for all fields.', 'error')
        return redirect(url_for('home'))
    except Exception as e:
        flash('An error occurred during prediction. Please try again.', 'error')
        print(f"Prediction error: {e}")
        return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)