from django.urls import path
from . import views



urlpatterns = [
    path('', views.DataWiPosListCreateView.as_view(), name="post_home"),
    path('<str:pk>/', views.PostRetrieveUpdateDeleteView.as_view(), name="post_detail"),
    path("datawipos/<str:pk>/", views.DatawiposPostRetrieveUpdateDeleteView.as_view(),name="data_wipos"),

]
