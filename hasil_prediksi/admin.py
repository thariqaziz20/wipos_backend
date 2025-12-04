from django.contrib import admin
from .models import HasilPrediksi
# Register your models here.


@admin.register(HasilPrediksi)
class DataWiPosAdmin(admin.ModelAdmin):
    list_display =["username","date","time", "lokasi",
                   "persentase_model_Random_Forest","persentase_model_SVM",
                    "persentase_model_knn_thariq","persentase_model_naive_bayes",
                        "persentase_model_knn_farrel","persentase_model_neural_network",
                            "lokasi_random_forest","lokasi_svm",
                                "lokasi_knn_thariq","lokasi_naive_bayes",
                                    "lokasi_knn_farrel","lokasi_neural_network"]
    list_filter = ['username',"lokasi"]