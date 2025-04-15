import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os
from django.conf import settings
from django.shortcuts import render
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


def home_view(request):
    return render(request,'home.html')

def train_model_view(request):
    # CSV path
    csv_path = os.path.join(settings.MEDIA_ROOT, 'balanced_crop_recommendation_dataset.csv')
    df = pd.read_csv(csv_path)

    # Split features and target
    X = df.drop('Label', axis=1)
    y = df['Label']

    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Pipeline creation
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    # Save model & encoder in media/
    model_path = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation_model.pkl')
    encoder_path = os.path.join(settings.MEDIA_ROOT, 'crop_label_encoder.pkl')
    joblib.dump(pipeline, model_path)
    joblib.dump(le, encoder_path)

    # Accuracy score
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Feature Importance Chart
    clf = pipeline.named_steps['clf']
    importance = clf.feature_importances_
    feature_names = X.columns

    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, importance, color='skyblue')
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()

    # Save chart in media/
    chart_path = os.path.join(settings.MEDIA_ROOT, 'feature_importance.png')
    plt.savefig(chart_path)
    plt.close()

    # Send relative path for rendering in template
    return render(request, 'train_result.html', {
        'accuracy': f"{accuracy:.4f}",
        'unique_labels': df['Label'].unique(),
        'chart_path': '/media/feature_importance.png'  # relative URL path
    })


def predict_crop_view(request):
    predicted_crop = None
    error_message = None

    if request.method == 'POST':
        try:
            # Load model and encoder
            model_path = os.path.join(settings.MEDIA_ROOT, 'crop_recommendation_model.pkl')
            encoder_path = os.path.join(settings.MEDIA_ROOT, 'crop_label_encoder.pkl')
            model = joblib.load(model_path)
            le = joblib.load(encoder_path)

            # Get input values
            values = [
                float(request.POST.get('N')),
                float(request.POST.get('P')),
                float(request.POST.get('K')),
                float(request.POST.get('temperature')),
                float(request.POST.get('humidity')),
                float(request.POST.get('ph')),
                float(request.POST.get('rainfall')),
            ]

            # Predict
            prediction = model.predict([values])
            predicted_crop = le.inverse_transform(prediction)[0]

        except Exception as e:
            error_message = f"Error: {str(e)}"

    return render(request, 'predict_result.html', {
        'predicted_crop': predicted_crop,
        'error_message': error_message
    })
