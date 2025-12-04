from django.contrib import admin
from .models import ModelWithPickle
# Register your models here.


@admin.register(ModelWithPickle)
class ModelWithPickleAdmin(admin.ModelAdmin):
    list_display =["username","date","time","svm_model","rf_model",
                   "knn_thariq_model","naive_bayes_model","knn_farrel_model",
                   "neural_network_model"
                   ]
    list_filter = ['username']