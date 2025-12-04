from django.contrib import admin
from .models import BackgroundService
# Register your models here.


@admin.register(BackgroundService)
class DataWiPosAdmin(admin.ModelAdmin):
    list_display =["username","date","time","ssid","macaddress","mackonversi", "rssi"]
    list_filter = ['username']