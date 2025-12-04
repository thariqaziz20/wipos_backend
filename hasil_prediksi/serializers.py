from rest_framework.serializers import ModelSerializer
from .models import HasilPrediksi


class HasilPrediksiSerializer(ModelSerializer):
    class Meta: 
        model = HasilPrediksi
        fields = '__all__'
