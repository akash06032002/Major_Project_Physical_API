from flask import Flask, request
from flask_restful import Resource, Api
import pickle
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
api = Api(app)

class MentalHealthPrediction(Resource):
    def get(self, steps=None, heart_rate=None, sleep_duration=None, stress=None):
        if all(param is not None for param in [steps, heart_rate, sleep_duration, stress]):
            # Path parameters provided, proceed with prediction
            try:
                # Convert the values to the appropriate data types
                steps = int(steps)
                heart_rate = int(heart_rate)
                sleep_duration = float(sleep_duration)
                stress = int(stress)

                # Add your model loading and prediction logic here
                # For demonstration purposes, we'll use a placeholder model
                # Replace the following lines with your actual model loading and prediction code
                model = pickle.load(open('model.pkl', 'rb'))
                features = [[1,steps, heart_rate, sleep_duration, stress]]
                prediction = model.predict(features)

                # Return the prediction as JSON
                return {"prediction": int(prediction[0])}

            except Exception as e:
                # Handle exceptions, such as invalid input
                return {"error": str(e)}, 400  # Return a 400 status code for bad requests
        else:
            # No specific path parameters provided
            return {"message": "Please provide values for Steps, HeartRate, SleepDuration, and Stress in the URL."}, 400

api.add_resource(MentalHealthPrediction, '/predict', '/predict/<int:steps>/<int:heart_rate>/<float:sleep_duration>/<int:stress>')

if __name__ == '__main__':
    # Run the app on a specific port (e.g., 5000)
    app.run(debug=True, port=5000)
