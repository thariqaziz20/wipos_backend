from django.contrib import admin
from .models import DataWiPos
# Register your models here.


@admin.register(DataWiPos)
class DataWiPosAdmin(admin.ModelAdmin):
    list_display =["username","date","time", "lokasi","ssid","macaddress","mackonversi", "rssi"]
    list_filter = ['username']