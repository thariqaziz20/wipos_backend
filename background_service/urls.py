from django.urls import path
from . import views



urlpatterns = [
    path('', views.BackgroundServiceListCreateView.as_view(), name="post_home"),
    path('<str:pk>/', views.PostRetrieveUpdateDeleteView.as_view(), name="post_detail"),
]
