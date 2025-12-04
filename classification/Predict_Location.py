import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from background_service.models import BackgroundService
from background_service.serializers import BackgroundServiceSerializer
from data_wipos.models import DataWiPos
from data_wipos.serializers import DataWiPosSerializer
from django_filters.rest_framework import DjangoFilterBackend
from datetime import datetime, timedelta

def PredictLocation(username):
    dataset = DataWiPos.objects.filter(username=username)
    queryset = BackgroundService.objects.filter(username=username)
    serializer = BackgroundServiceSerializer(queryset, many=True)
    last_data = queryset.last()

    if last_data:
        last_date = last_data.date
        last_time = last_data.time
        # print("last_date", last_date)
        # print(last_time)

        date = last_date
        time = last_time
        
        # Kurangi 1 menit dari last_time, tetap dengan date yang sama
        time_1_min_ago = (datetime.combine(date, time) - timedelta(minutes=1)).time()
        
        # print("Last time:", time)
        # print("Time 1 minute ago:", time_1_min_ago)

        queryset_filtered = BackgroundService.objects.filter(username=username, date=date, time__range=(time_1_min_ago, time))
        mac_konversi_data = queryset_filtered.values_list('mackonversi', flat=True)
        rssi_data = queryset_filtered.values_list('rssi', flat=True)
        merged_data = np.column_stack((mac_konversi_data, rssi_data))
        # print("BackgroundService:", merged_data.shape)

        # Mengambil data dari Dataset training dari "username"
        training_dataset = dataset.values_list('mackonversi', 'rssi')
        # print("Training dataset shape:", np.array(training_dataset).shape)
        
        # Mengambil MAC address unik dari training dataset
        training_mac_addresses = set(dataset.values_list('mackonversi', flat=True).distinct())
        # print("Unique MAC addresses in training dataset:", len(training_mac_addresses))
        
        # Memisahkan merged_data berdasarkan MAC address
        # Data yang MAC address-nya ada di training dataset
        data_found = []
        # Data yang MAC address-nya tidak ada di training dataset
        data_not_found = []
        
        for mac, rssi in merged_data:
            if mac in training_mac_addresses:
                data_found.append([mac, rssi])
            else:
                data_not_found.append([mac, rssi])
        
        data_found = np.array(data_found) if data_found else np.array([]).reshape(0, 2)
        data_not_found = np.array(data_not_found) if data_not_found else np.array([]).reshape(0, 2)
        
        # print(f"Data found in training dataset: {data_found.shape}")
        # print(f"Data NOT found in training dataset: {data_not_found.shape}")
        
        # Menampilkan data
        serialized_data = serializer.data
        return merged_data, serialized_data, data_found, data_not_found
    else:
        return None, None
    
