from . import views
from django.urls import path


urlpatterns = [
    path("signup/", views.SignUpView.as_view(), name="signup"),
    path("login/", views.Loginview.as_view(),name="login"),
    path("login/<str:email>/", views.LoginGetData.as_view(),name="loginData"),

]
