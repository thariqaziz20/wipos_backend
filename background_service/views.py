from rest_framework.decorators import api_view, APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status, generics, mixins, viewsets
from rest_framework.permissions import IsAuthenticated
from .serializers import BackgroundServiceSerializer
from .models import BackgroundService
from django.shortcuts import get_object_or_404

# Create your views here.
class BackgroundServiceListCreateView(APIView):
    serializer_class = BackgroundServiceSerializer
    # permission_classes =[IsAuthenticated]

    """
    a view for creating and listing data
    """
    def get(self, request:Request, *args, **kwargs):
        posts = BackgroundService.objects.all()
        serializer = self.serializer_class(instance=posts, many=True)
        response = {
                "data" : serializer.data
            }
        return Response(data=response, status=status.HTTP_200_OK)

    def post(self, request : Request, *args, **kwargs):
        data = request.data
        serializer = self.serializer_class(data=data)
        
        username = data.get('username', 'Unknown')
        print(f"Username: {username} | Jumlah dataset untuk Prediksi yang dimasukkan: {len(data)}")
        
        if serializer.is_valid():
            serializer.save()
            
            response = {
                "message" : "data",
                "data" : serializer.data
            }

            return Response(data=response, status=status.HTTP_201_CREATED)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        



class PostRetrieveUpdateDeleteView(APIView):
    serializer_class = BackgroundServiceSerializer
    # permission_classes =[IsAuthenticated]


    def get(self, requesr : Request, pk:int):
        post = get_object_or_404(BackgroundService, pk=pk)
        serializer = self.serializer_class(instance=post)

        return Response(data=serializer.data, status=status.HTTP_200_OK)
    
    def put(self, request:Request, pk:int):
        post = get_object_or_404(BackgroundService, pk=pk)

        data = request.data
        serializer = self.serializer_class(instance=post,
            data = data                                   
        )
        if serializer.is_valid():
            serializer.save()

            response={
                "message" : "Post Updated",
                "data" : serializer.data
            }

            return Response(data=response, status=status.HTTP_200_OK)
        
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    def delete(self, request:Request, pk:int):
        post = get_object_or_404(BackgroundService, pk=pk)

        post.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

