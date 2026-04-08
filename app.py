from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and scaler once at startup
model  = pickle.load(open('model.pkl',  'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        iq   = float(request.form['feature1'])
        cgpa = float(request.form['feature2'])

        # Note: model was trained with columns [cgpa, iq]
        user_input        = np.array([[cgpa, iq]])
        user_input_scaled = scaler.transform(user_input)

        result = model.predict(user_input_scaled)[0]

        if result == 1:
            prediction = "✅ Likely to be PLACED!"
            color = "#28a745"
        else:
            prediction = "❌ Not likely to be Placed"
            color = "#dc3545"

        return render_template('index.html',
                               prediction=prediction,
                               color=color,
                               iq=int(iq),
                               cgpa=cgpa)

    except Exception as e:
        return render_template('index.html',
                               prediction=f"Error: {str(e)}",
                               color="#f0ad4e")

if __name__ == '__main__':
    app.run(debug=True)
