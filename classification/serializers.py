from rest_framework.serializers import ModelSerializer
from .models import ModelWithPickle

class ModelWithPickleSerializer(ModelSerializer):
    class Meta: 
        model = ModelWithPickle
        fields = '__all__'
