from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('datawipos/', include("data_wipos.urls")),
    path('backgroundservice/', include("background_service.urls")),
    path('hasil/', include("hasil_prediksi.urls")),
    path('classification/', include("classification.urls")),
    path("auth/", include("accounts.urls")),
]