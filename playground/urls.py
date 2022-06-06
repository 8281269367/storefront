from pydoc import plain
from django.urls import URLPattern, path,include
from . import views

urlpatterns = [
    path('hello/', views.say_hello)

]