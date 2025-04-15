# Crop Recommendation System using ML

This project is a machine learning-based crop recommendation system. It recommends the best crop to grow based on various environmental and soil factors. The model is built using the Random Forest Classifier and trained on a crop recommendation dataset.

## Project Structure:

- **train_model_view**: This view is used to train the machine learning model. It reads the dataset, trains a Random Forest Classifier, and visualizes feature importance.
- **predict_crop_view**: This view allows users to input environmental factors and receive a predicted crop recommendation.
- **media/**: Contains files like the dataset (`balanced_crop_recommendation_dataset.csv`) and model files.
- **templates/**: Contains HTML templates for the frontend.
- **static/**: Contains CSS files for styling the HTML templates.

## Requirements:

- Python 3.x
- Django
- scikit-learn
- matplotlib
- joblib
- pandas

## Installation:

1. Clone the repository or download the source code.
2. Install the required packages
3. Make migrations and migrate the database: python manage.py makemigrations, python manage.py migrate
4. Run the Django server: python manage.py runserver
   Access the web application on `http://127.0.0.1:8000`.

## Usage:

- To train the model, visit `/train_model/` on the app.
- To predict a crop, visit `/predict_crop/` and input the environmental factors (N, P, K, temperature, humidity, pH, and rainfall).

## Author:

KURUVA BHANU PRAKASH 
kuruvabhanu28prakash@gmail.com
9014663588


