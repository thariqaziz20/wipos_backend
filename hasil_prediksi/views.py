from rest_framework.decorators import api_view, APIView
from rest_framework.response import Response
from rest_framework.request import Request
from rest_framework import status, generics, mixins, viewsets
from rest_framework.permissions import IsAuthenticated
from .serializers import HasilPrediksiSerializer
from .models import HasilPrediksi
from data_wipos.serializers import DataWiPosSerializer
from data_wipos.models import DataWiPos
from django.shortcuts import get_object_or_404

# Create your views here.
class DataWiPosListCreateView(APIView):
    serializer_class = HasilPrediksiSerializer
    # permission_classes =[IsAuthenticated]

    """
    a view for creating and listing data
    """
    def get(self, request:Request, *args, **kwargs):
        posts = HasilPrediksi.objects.all()
        serializer = self.serializer_class(instance=posts, many=True)
        response = {
                "data" : serializer.data
            }
        return Response(data=response, status=status.HTTP_200_OK)

    def post(self, request : Request, *args, **kwargs):
        data = request.data
        serializer = self.serializer_class(data=data)
        print(serializer)
        if serializer.is_valid():
            serializer.save()
            
            response = {
                "message" : "data",
                "data" : serializer.data
            }

            return Response(data=response, status=status.HTTP_201_CREATED)
        return Response(data=serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        



class PostRetrieveUpdateDeleteView(APIView):
    serializer_class = HasilPrediksiSerializer
    # permission_classes =[IsAuthenticated]


    def get(self, requesr : Request, pk:int):
        post = get_object_or_404(HasilPrediksi, pk=pk)
        serializer = self.serializer_class(instance=post)

        return Response(data=serializer.data, status=status.HTTP_200_OK)
    
    def put(self, request:Request, pk:int):
        post = get_object_or_404(HasilPrediksi, pk=pk)

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
        post = get_object_or_404(HasilPrediksi, pk=pk)

        post.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)
    



class DatawiposPostRetrieveUpdateDeleteView(APIView):
    serializer_class = DataWiPosSerializer
    permission_classes =[IsAuthenticated]


    def get(self, requesr : Request, pk:int):
        post = get_object_or_404(DataWiPos, pk=pk)
        serializer = self.serializer_class(instance=post)

        return Response(data=serializer.data, status=status.HTTP_200_OK)
    
    def put(self, request:Request, pk:int):
        post = get_object_or_404(DataWiPos, pk=pk)

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
        post = get_object_or_404(DataWiPos, pk=pk)

        post.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

