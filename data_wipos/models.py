from django.db import models

# Create your models here.

class DataWiPos(models.Model):
    username        = models.CharField(max_length=20)
    date            = models.DateField(auto_now=False, auto_now_add=True)
    time            = models.TimeField(auto_now=False, auto_now_add=True)
    lokasi          = models.CharField(max_length=50)
    ssid            = models.CharField(max_length=50)
    macaddress      = models.CharField(max_length=18)
    mackonversi     = models.CharField(max_length=50)
    rssi            = models.IntegerField()
