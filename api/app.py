from flask import Flask, request, jsonify
from model_loader import load_model

app = Flask(__name__)
model = load_model()

@app.route("/")
def home():
    return {"message": "API ML operational"}

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]  # Ex: [5.1, 3.5, 1.4, 0.2]

        prediction = model.predict([data])[0]

        return jsonify({
            "input": data,
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
