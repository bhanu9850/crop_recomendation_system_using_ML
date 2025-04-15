from django.urls import path
from . import views

urlpatterns = [
    path("",views.home_view,name = "home"),
    path('train/', views.train_model_view, name='train_model'),
    path('predict/', views.predict_crop_view, name='predict_crop'),
]
