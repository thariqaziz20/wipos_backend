from rest_framework.serializers import ModelSerializer
from .models import BackgroundService


class BackgroundServiceSerializer(ModelSerializer):
    class Meta: 
        model = BackgroundService
        fields = '__all__'
