from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('receive_path/', views.receive_path, name='receive_path'),
]
