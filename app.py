from flask import Flask, request, redirect
from flask_restful import Resource, Api
import joblib
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    CORS(app)
    api = Api(app)

    @app.route('/')
    def home():
        return redirect('/predict')

    class MentalHealthPrediction(Resource):
        def get(self, steps=None, heart_rate=None, sleep_duration=None, stress=None):
            # Check if parameters are provided in the URL path
            if all(param is not None for param in [steps, heart_rate, sleep_duration, stress]):
                # Parameters provided in the URL path, proceed with prediction
                try:
                    # Convert the values to the appropriate data types
                    steps = int(steps)
                    heart_rate = int(heart_rate)
                    sleep_duration = float(sleep_duration)
                    stress = int(stress)

                except ValueError as e:
                    return {"error": f"Invalid value provided: {e}"}, 400

            # If not all parameters are provided in the URL path, check query parameters
            else:
                # Access all query parameters as a dictionary
                params = request.args.to_dict()

                if all(param in params for param in ['steps', 'heart_rate', 'sleep_duration', 'stress']):
                    # Parameters provided in the query parameters, proceed with prediction
                    try:
                        # Convert the values to the appropriate data types
                        steps = int(params['steps'])
                        heart_rate = int(params['heart_rate'])
                        sleep_duration = float(params['sleep_duration'])
                        stress = int(params['stress'])

                    except ValueError as e:
                        return {"error": f"Invalid value provided: {e}"}, 400

                else:
                    # Not all required parameters provided
                    return {"message": "Please provide values for Steps, HeartRate, SleepDuration, and Stress. \n /predict?steps=7000&heart_rate=78&sleep_duration=9&stress=1 - in this manner"}, 400

            # Add your model loading and prediction logic here
            # For demonstration purposes, we'll use a placeholder model
            # Replace the following lines with your actual model loading and prediction code
            model = joblib.load('model.pkl')  # Use joblib instead of pickle
            joblib.dump(model, 'model.pkl', protocol=0)
            features = [[1, steps, heart_rate, sleep_duration, stress]]
            prediction = model.predict(features)

            # Return the prediction as JSON
            return {"prediction": int(prediction[0])}

    api.add_resource(MentalHealthPrediction, '/predict', '/predict/<int:steps>/<int:heart_rate>/<float:sleep_duration>/<int:stress>')
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, port=5000)
