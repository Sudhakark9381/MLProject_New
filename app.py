
from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Create Flask application
application = Flask(__name__)
app = application


# Home page route
@app.route('/')
def index():
    return render_template('index.html')


# Prediction page route
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    # Display the form
    if request.method == 'GET':
        return render_template('home.html')

    try:
        # Read input values from the HTML form
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get(
                'parental_level_of_education'
            ),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get(
                'test_preparation_course'
            ),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score'))
        )

        # Convert the input into a DataFrame
        pred_df = data.get_data_as_data_frame()

        print("Input data:")
        print(pred_df)
        print("Before Prediction")

        # Create the prediction pipeline
        predict_pipeline = PredictPipeline()

        print("Mid Prediction")

        # Generate prediction
        results = predict_pipeline.predict(pred_df)

        print("After Prediction")

        # Display prediction result
        return render_template(
            'home.html',
            results=results[0]
        )

    except Exception as e:
        # Print the complete error in the terminal
        app.logger.exception("Prediction failed")

        # Display a message on the webpage
        return render_template(
            'home.html',
            error=f"Prediction failed: {str(e)}"
        ), 500


# Start Flask server
# if __name__ == "__main__":
#     app.run(host="127.0.0.1", port=5000, debug=False)

if __name__ == "__main__":
    print("Starting ML prediction application...")

    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False,
        use_reloader=False
    )
