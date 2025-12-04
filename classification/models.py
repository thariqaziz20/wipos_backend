from django.db import models

# Create your models here.

class ModelWithPickle(models.Model):
    username = models.CharField(max_length=100)
    date     = models.DateField(auto_now=False, auto_now_add=True)
    time     = models.TimeField(auto_now=False, auto_now_add=True)
    svm_model = models.BinaryField()
    rf_model = models.BinaryField()
    knn_thariq_model = models.BinaryField()
    naive_bayes_model = models.BinaryField()
    knn_farrel_model = models.BinaryField()
    neural_network_model = models.BinaryField()
    
    # Kolom akurasi untuk 6 model
    akurasi_svm = models.FloatField(null=True, blank=True, default=None)
    akurasi_rf = models.FloatField(null=True, blank=True, default=None)
    akurasi_knn_thariq = models.FloatField(null=True, blank=True, default=None)
    akurasi_naive_bayes = models.FloatField(null=True, blank=True, default=None)
    akurasi_knn_farrel = models.FloatField(null=True, blank=True, default=None)
    akurasi_neural_network = models.FloatField(null=True, blank=True, default=None)
