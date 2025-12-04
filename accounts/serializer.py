from rest_framework import serializers
from rest_framework.serializers import ModelSerializer
from rest_framework.validators import ValidationError
from rest_framework.authtoken.models import Token
from .models import User


class SingUpSerializer(serializers.ModelSerializer):
    email = serializers.CharField(max_length=80)
    username = serializers.CharField(max_length=45)
    password = serializers.CharField(min_length=4,write_only=True)

    class Meta:
        model = User
        fields = ["email", "username", "password"]

    def validate(self, attrs):
        email_exists = User.objects.filter(email=attrs['email']).exists()

        if email_exists:
            raise ValidationError("Email has already been used")
        
        return super().validate(attrs)
    
    def create(self, validate_data):
        password = validate_data.pop("password")

        user = super().create(validate_data)

        user.set_password(password)

        user.save()
        
        Token.objects.create(user=user)

        return user
    
class UserSerializer(ModelSerializer):
    class Meta:
        model = User
        fields = '__all__'