from django.urls import path
from .views import index, process_text
  
urlpatterns = [
    path('', index),
    path('process_text', process_text)
]