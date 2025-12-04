from rest_framework.serializers import ModelSerializer
from .models import DataWiPos


class DataWiPosSerializer(ModelSerializer):
    class Meta: 
        model = DataWiPos
        fields = '__all__'
