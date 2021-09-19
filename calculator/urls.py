from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='calculator'),
    path('iv/', views.implied_volatility, name='implied-volatility'),
]