from django.db import models

# Create your models here.

class HasilPrediksi(models.Model):
    username                            = models.CharField(max_length=20)
    date                                = models.DateField(auto_now=False, auto_now_add=True)
    time                                = models.TimeField(auto_now=False, auto_now_add=True)
    lokasi                              = models.CharField(max_length=50, default='')
    persentase_model_Random_Forest      = models.CharField(max_length=50, default='')
    persentase_model_SVM                = models.CharField(max_length=50, default='')
    persentase_model_knn_thariq         = models.CharField(max_length=50, default='')
    persentase_model_naive_bayes         = models.CharField(max_length=50, default='')
    persentase_model_knn_farrel         = models.CharField(max_length=50, default='')
    persentase_model_neural_network      = models.CharField(max_length=50, default='')
    lokasi_random_forest            = models.CharField(max_length=50, default='')
    lokasi_svm                      = models.CharField(max_length=50, default='')
    lokasi_knn_thariq               = models.CharField(max_length=50, default='')
    lokasi_naive_bayes               = models.CharField(max_length=50, default='')
    lokasi_knn_farrel               = models.CharField(max_length=50, default='')
    lokasi_neural_network          = models.CharField(max_length=50, default='')
    persentase_lokasi_tidak_ditemukan = models.CharField(max_length=50, default='')





