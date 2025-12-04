from django.shortcuts import render
from django.contrib.auth import authenticate
from .serializer import SingUpSerializer
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework.views import APIView
from .serializer import UserSerializer
from .models import User
from django.shortcuts import get_object_or_404

# Create your views here.


class SignUpView(generics.GenericAPIView):
    serializer_class = SingUpSerializer
    print(serializer_class)
    def post(self,request:Request):
        data = request.data

        serializer=self.serializer_class(data=data)

        if serializer.is_valid():
            serializer.save()

            response={
                "message":"Username Created Successfully",
                "data":serializer.data
            }

            return Response(data=response, status=status.HTTP_201_CREATED)
        
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    






class Loginview(APIView):
    serializer_class = SingUpSerializer

    def post(self,request:Request):
        email=request.data.get("email")
        password=request.data.get("password")

        user=authenticate(email=email,password=password)
        
        
        if user is not None:
            post = get_object_or_404(User, email=email)
            serializer = self.serializer_class(instance=post)
            variable1_value = serializer.data.get('username')
            variable2_value = serializer.data.get('email')

            response={
                "message":"Login Successfull",
                "username": variable1_value,
                "email": variable2_value,
                "token":user.auth_token.key,
            }
            return Response(data=response,status=status.HTTP_200_OK)
        
        else:
            return Response(data={"message":"invalid username or password"})

    def get(self,request:Request):
        content={
            "user":str(request.user),
            "auth":str(request.auth)
        }

        return Response(data=content, status=status.HTTP_200_OK)

class LoginGetData(APIView):
    serializer_class = SingUpSerializer
    def get(self, request:Request, email):
        post = get_object_or_404(User, email=email)
        serializer = self.serializer_class(instance=post)
        response={
            "data" : serializer.data
        }
        return Response(data=response, status=status.HTTP_200_OK)








"""
class LoginView(APIView):
    
    def post(self,request:Request):
        email = request.data.get("email")
        password = request.data.get("password")

        user=authenticate(email=email, password=password)

        if user is not None:
            response={
                "message" : "Login Successfull",
                "token" : user.auth_token.key
            }

            return Response(data=response, status=status.HTTP_200_OK)
        
        else:
            return Response(data={"message":"Invalid Email or Password"})
        
        
        pass
        
        

    def get(self,request:Request):
        content={"user":str(request.user),"auth":str(request.auth)}

        return Response(data=content, status=status.HTTP_200_OK)

        """

# class SignUpView(generics.GenericAPIView):
#     serializer_class=SingUpSerializer

#     def post(self,request:Request):
#         data=request.data
    
#         serializer = self.serializer_class(data=data)

#         if serializer.is_valid():
#             serializer.save()

#             response={
#                 "message":"Username Created Successfully",
#                 "data" : serializer.data
#             }
#             return Response(data=response, status=status.HTTP_201_CREATED)
        
#         return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)