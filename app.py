from flask import Flask, request, jsonify, render_template
from predict import predict_alphabet
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return "Please use POST method to submit a file for prediction."
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        if file:
            filename = file.filename
            
            if not os.path.exists("uploads/"):
                os.makedirs("uploads/")
                
            filepath = os.path.join("uploads/", filename)
            try:
                file.save(filepath)
                result = predict_alphabet(filepath)
            except Exception as e:
                return jsonify({"error": str(e)}), 500
            finally:
                os.remove(filepath)  # Clean up the file after prediction
            return jsonify({"prediction": result})
    
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
