from rest_framework import generics
from django_filters.rest_framework import DjangoFilterBackend
from data_wipos.serializers import DataWiPosSerializer
from data_wipos.models import DataWiPos
from rest_framework.response import Response
from rest_framework import status
import pandas as pd 
import numpy as np
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from .Model_Klasifikasi import ModelKlasifikasi
from .models import ModelWithPickle
from .serializers import ModelWithPickleSerializer
from collections import Counter
from .Predict_Location import PredictLocation
from hasil_prediksi.models import HasilPrediksi
from hasil_prediksi.serializers import HasilPrediksiSerializer
from rest_framework.decorators import api_view
from rest_framework.request import Request


@api_view(['GET'])
def accuracy_report(request: Request):
    username = request.query_params.get('username')
    if not username:
        return Response(
            {"detail": "query parameter 'username' is required"},
            status=status.HTTP_400_BAD_REQUEST
        )
    
    try:
        # Ambil HANYA record terbaru (latest) untuk username tersebut
        latest_record = ModelWithPickle.objects.filter(username__iexact=username).latest('date', 'time')
        
        if not latest_record:
            return Response(
                {"detail": f"No model data found for username '{username}'"},
                status=status.HTTP_404_NOT_FOUND
            )
        
        # Format response dengan akurasi semua model dari record terbaru saja
        response_data = {
            'id': latest_record.id,
            'username': latest_record.username,
            'date': latest_record.date,
            'time': latest_record.time,
            'accuracies': {
                'svm': latest_record.akurasi_svm,
                'random_forest': latest_record.akurasi_rf,
                'knn_thariq': latest_record.akurasi_knn_thariq,
                'naive_bayes': latest_record.akurasi_naive_bayes,
                'knn_farrel': latest_record.akurasi_knn_farrel,
                'neural_network': latest_record.akurasi_neural_network,
            }
        }
        
        return Response(response_data, status=status.HTTP_200_OK)
    
    except ModelWithPickle.DoesNotExist:
        return Response(
            {"detail": f"No model data found for username '{username}'"},
            status=status.HTTP_404_NOT_FOUND
        )
    except Exception as e:
        return Response(
            {"detail": f"Error retrieving accuracy data: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class FittingModel(generics.ListAPIView):
    queryset = DataWiPos.objects.all()
    serializer_class = DataWiPosSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['username']

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        data = serializer.data

        if self.check_different_values(data):
            
            return Response(data=data, status=status.HTTP_200_OK)

        data_name = data[0] if data else None
        username = data_name.get('username') if data_name else None

        hasil = ModelKlasifikasi(username, data)
        response = {
            "Status" : hasil[4],
            "Hasil Model Klasifikasi SVM": hasil[0],
            "Hasil Model Klasifikasi RF": hasil[1],
            "Hasil Model KNN Thariq" : hasil[2],
            "Hasil Model Naive Bayes" : hasil[3]
        }
        print(response)
        return Response(data=response, status=status.HTTP_200_OK)

    # Fungsi untuk memeriksa apakah ada nilai yang berbeda dalam data kolom pertama
    def check_different_values(self, data):
        # Jika tidak ada data, tidak ada nilai yang berbeda
        if not data:
            return False

        # Ambil nilai dari kolom pertama
        first_column_values = [item['username'] for item in data]

        # Jika ada nilai yang berbeda dalam data kolom pertama, return True
        if len(set(first_column_values)) > 1:
            return True

        return False



class PredictModel(generics.ListAPIView):
    queryset = ModelWithPickle.objects.all()
    serializer_class = ModelWithPickleSerializer
    filter_backends = [DjangoFilterBackend]
    filterset_fields = ['username']

    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())
        serializer = self.get_serializer(queryset, many=True)
        data = serializer.data

        if self.check_different_values(data):
            
            return Response(data=data, status=status.HTTP_200_OK)

        data_name = data[0] if data else None
        username = data_name.get('username') if data_name else None

        # // Mengambil model dari database
        data_name = data[0] if data else None
        username = data_name.get('username') if data_name else None

        # Use filter + order_by + first to avoid MultipleObjectsReturned
        model_with_pickle_instance = ModelWithPickle.objects.filter(username=username).order_by('-id').first()
        if model_with_pickle_instance is None:
            return Response(data={"message": "No model found for the given username."}, status=status.HTTP_404_NOT_FOUND)

        svm_model = pickle.loads(model_with_pickle_instance.svm_model)
        rf_model = pickle.loads(model_with_pickle_instance.rf_model)
        knn_thariq_model = pickle.loads(model_with_pickle_instance.knn_thariq_model)
        naive_bayes_model = pickle.loads(model_with_pickle_instance.naive_bayes_model)
        
        def make_predictions(svm_model, rf_model, knn_thariq_model, naive_bayes_model, prediction_data):
            # Bypass jika tidak ada data untuk diprediksi
            if prediction_data is None or len(prediction_data) == 0:
                print("Data Prediksi tidak ada pada Data Training, " \
                "kosong, bypass semua prediksi. Lokasi Tidak Ditemukan.")
                return np.array([]), np.array([]), np.array([]), np.array([])
            
            print("Making predictions on data of shape:", prediction_data.shape)
            
            # Konversi prediction_data ke DataFrame untuk Naive Bayes (karena model dilatih dengan DataFrame)
            input_df = pd.DataFrame(prediction_data, columns=["mackonversi", "rssi"])
            
            svm_prediction = svm_model.predict(prediction_data)
            rf_prediction = rf_model.predict(prediction_data)
            knn_thariq_prediction = knn_thariq_model.predict(prediction_data)
            naive_bayes_prediction = naive_bayes_model.predict(input_df)
            
            return svm_prediction, rf_prediction, knn_thariq_prediction, naive_bayes_prediction
        
        input_data, background_data, mac_found, mac_not_found = PredictLocation(username)
        print("Input Data shape:", input_data.shape)
        print("MAC Found shape:", mac_found.shape)
        print("MAC Not Found shape:", mac_not_found.shape)

        if input_data is None or background_data is None:
            # Construct failure response
            failure_response = {
                "message": "Failed to retrieve input data or background data for the given username."
            }
            return Response(data=failure_response, status=status.HTTP_400_BAD_REQUEST)

        # Jika input_data kosong, langsung kembalikan lokasi tidak ditemukan
        if input_data is None or len(input_data) == 0:
            lokasi = "Lokasi Tidak Ditemukan"
            response = {"data": {
                "Username": username,
                "Dominan Result": {"Model": "notfound", "Percentage": "100.00 %", "Final Location": lokasi}
            }}
            return Response(data=response, status=status.HTTP_200_OK)

        # Make prediction
        svm_prediction, rf_prediction, knn_thariq_prediction, naive_bayes_prediction = make_predictions(svm_model, rf_model, knn_thariq_model, naive_bayes_model, mac_found)
        
        # Hitung persentase mac found dan not found terhadap input data
        not_found = len(mac_not_found)
        total_input_data = len(input_data)
        
        persentase_not_found = (not_found / total_input_data * 100) if total_input_data > 0 else 0
        persentase_not_found_str = "{:.2f} %".format(persentase_not_found)
        print(f"MAC Not Found: {not_found} ({persentase_not_found_str})")
        
        # Cek apakah mac_found kosong (tidak ada data untuk diprediksi)
        if len(mac_found) == 0:
            # Jika tidak ada data untuk diprediksi, set semua lokasi menjadi "Lokasi Tidak Ditemukan"
            lokasi_svm_prediction = "Lokasi Tidak Ditemukan"
            lokasi_rf_prediction = "Lokasi Tidak Ditemukan"
            lokasi_knn_thariq_prediction = "Lokasi Tidak Ditemukan"
            lokasi_naive_bayes_prediction = "Lokasi Tidak Ditemukan"
            
            persentase_max_svm_prediction = "100.00 %"
            persentase_max_rf_prediction = "100.00 %"
            persentase_max_knn_thariq_prediction = "100.00 %"
            persentase_max_naive_bayes_prediction = "100.00 %"
            
            persentase_SVM = {}
            persentase_RF = {}
            persentase_KNN_Thariq = {}
            persentase_Naive_Bayes = {}
        else:
            # Hitung kemunculan setiap string dalam array untuk semua model
            counts_SVM = Counter(svm_prediction)
            counts_RF = Counter(rf_prediction)
            counts_KNN_Thariq = Counter(knn_thariq_prediction)
            counts_Naive_Bayes = Counter(naive_bayes_prediction)

            # Hitung total kemunculan untuk setiap model
            total_SVM = sum(counts_SVM.values())
            total_RF = sum(counts_RF.values())
            total_KNN_Thariq = sum(counts_KNN_Thariq.values())
            total_Naive_Bayes = sum(counts_Naive_Bayes.values())

            # Hitung dan cetak persentase setiap elemen untuk model SVM
            persentase_SVM = {}
            for nilai, jumlah in counts_SVM.items():
                persentase = (jumlah / total_SVM) * 100
                persentase_SVM[nilai] = "{:.2f} %".format(persentase)

            # Hitung dan cetak persentase setiap elemen untuk model RF
            persentase_RF = {}
            for nilai, jumlah in counts_RF.items():
                persentase = (jumlah / total_RF) * 100
                persentase_RF[nilai] = "{:.2f} %".format(persentase)

            # Hitung dan cetak persentase setiap elemen untuk model KNN Thariq
            persentase_KNN_Thariq = {}
            for nilai, jumlah in counts_KNN_Thariq.items():
                persentase = (jumlah / total_KNN_Thariq) * 100
                persentase_KNN_Thariq[nilai] = "{:.2f} %".format(persentase)

            # Hitung dan cetak persentase setiap elemen untuk model Naive Bayes
            persentase_Naive_Bayes = {}
            for nilai, jumlah in counts_Naive_Bayes.items():
                persentase = (jumlah / total_Naive_Bayes) * 100
                persentase_Naive_Bayes[nilai] = "{:.2f} %".format(persentase)

            # Dapatkan hasil prediksi Ruangan dengan persentase terbesar
            lokasi_svm_prediction = max(counts_SVM, key=counts_SVM.get) if counts_SVM else None
            lokasi_rf_prediction = max(counts_RF, key=counts_RF.get) if counts_RF else None
            lokasi_knn_thariq_prediction = max(counts_KNN_Thariq, key=counts_KNN_Thariq.get) if counts_KNN_Thariq else None
            lokasi_naive_bayes_prediction = max(counts_Naive_Bayes, key=counts_Naive_Bayes.get) if counts_Naive_Bayes else None

            # Hitung persentase untuk hasil prediksi terbesar
            persentase_max_svm_prediction = "{:.2f} %".format((counts_SVM[lokasi_svm_prediction] / total_input_data) * 100) if total_SVM != 0 and lokasi_svm_prediction else "0.00 %"
            persentase_max_rf_prediction = "{:.2f} %".format((counts_RF[lokasi_rf_prediction] / total_input_data) * 100) if total_RF != 0 and lokasi_rf_prediction else "0.00 %"
            persentase_max_knn_thariq_prediction = "{:.2f} %".format((counts_KNN_Thariq[lokasi_knn_thariq_prediction] / total_input_data) * 100) if total_KNN_Thariq != 0 and lokasi_knn_thariq_prediction else "0.00 %"
            persentase_max_naive_bayes_prediction = "{:.2f} %".format((counts_Naive_Bayes[lokasi_naive_bayes_prediction] / total_input_data) * 100) if total_Naive_Bayes != 0 and lokasi_naive_bayes_prediction else "0.00 %"
        
        # Lokasi DOminan    
        def insert_hasil_prediksi_to_database(username, lokasi, 
                                          persentase_max_rf_prediction, persentase_max_svm_prediction, 
                                          persentase_max_knn_thariq_prediction, lokasi_svm_prediction, 
                                          lokasi_rf_prediction, lokasi_knn_thariq_prediction, 
                                          lokasi_naive_bayes_prediction, persentase_max_naive_bayes_prediction,
                                          persentase_not_found_str):
            # Buat instance serializer dengan data yang ingin dimasukkan
            data = {
                'username': username,
                'lokasi': lokasi,
                'lokasi_svm': lokasi_svm_prediction,
                'lokasi_random_forest': lokasi_rf_prediction,
                'lokasi_knn_thariq': lokasi_knn_thariq_prediction,
                'lokasi_naive_bayes': lokasi_naive_bayes_prediction,

                'persentase_model_Random_Forest': persentase_max_rf_prediction,
                'persentase_model_SVM': persentase_max_svm_prediction,
                'persentase_model_knn_thariq': persentase_max_knn_thariq_prediction,
                'persentase_model_naive_bayes': persentase_max_naive_bayes_prediction,
                'persentase_lokasi_tidak_ditemukan': persentase_not_found_str,
            }
            serializer = HasilPrediksiSerializer(data=data)

            # Periksa apakah data valid
            if serializer.is_valid():
                # Simpan data ke dalam database
                serializer.save()
                print("Data berhasil dimasukkan ke dalam database.")
            else:
                # Jika data tidak valid, cetak pesan kesalahan
                print("Gagal memasukkan data ke dalam database. Kesalahan:", serializer.errors)
                
        # ========================================================================================================
        # Normalize percentage values to numeric floats so we can compare them
        def to_float_percent(val):
            if isinstance(val, str):
                try:
                    return float(val.replace('%', '').strip())
                except Exception:
                    return 0.0
            try:
                return float(val)
            except Exception:
                return 0.0

        persentase_all = {
            "svm": to_float_percent(persentase_max_svm_prediction),
            "rf": to_float_percent(persentase_max_rf_prediction),
            "knn_thariq": to_float_percent(persentase_max_knn_thariq_prediction),
            "naive_bayes": to_float_percent(persentase_max_naive_bayes_prediction),
            "notfound": to_float_percent(persentase_not_found)
        }

        # Determine which model (or notfound) has the highest percentage
        model_tertinggi = max(persentase_all, key=persentase_all.get)
        persentase_tertinggi = "{:.2f} %".format(persentase_all[model_tertinggi])

        if model_tertinggi == "notfound":
            lokasi = "Lokasi Tidak Ditemukan"
        elif model_tertinggi == "svm":
            lokasi = lokasi_svm_prediction
        elif model_tertinggi == "rf":
            lokasi = lokasi_rf_prediction
        elif model_tertinggi == "knn_thariq":
            lokasi = lokasi_knn_thariq_prediction
        elif model_tertinggi == "naive_bayes":
            lokasi = lokasi_naive_bayes_prediction

        lokasi = lokasi
        insert_hasil_prediksi_to_database(username, lokasi, 
                                          persentase_max_rf_prediction, persentase_max_svm_prediction, 
                                          persentase_max_knn_thariq_prediction, lokasi_svm_prediction, 
                                          lokasi_rf_prediction, lokasi_knn_thariq_prediction,
                                          lokasi_naive_bayes_prediction, persentase_max_naive_bayes_prediction,
                                          persentase_not_found_str)

        # Buat respons
        response = {"data":{
            "Username": username,
            "Not Found": {"Lokasi": "Lokasi Tidak Ditemukan", 
                                       "Percentage": persentase_not_found_str},
            "SVM Result": persentase_SVM,
            "Max SVM Prediction": {"Lokasi": lokasi_svm_prediction, 
                                   "Percentage": persentase_max_svm_prediction},

            "RF Result": persentase_RF,
            "Max RF Prediction": {"Lokasi": lokasi_rf_prediction, 
                                  "Percentage": persentase_max_rf_prediction},

            "KNN Thariq Result": persentase_KNN_Thariq,
            "Max KNN Thariq Prediction": {"Lokasi": lokasi_knn_thariq_prediction, 
                                          "Percentage": persentase_max_knn_thariq_prediction},

            "Naive Bayes Result": persentase_Naive_Bayes,
            "Max Naive Bayes Prediction": {"Lokasi": lokasi_naive_bayes_prediction, 
                                          "Percentage": persentase_max_naive_bayes_prediction},

            "Dominan Result": {"Model": model_tertinggi, 
                               "Percentage": persentase_tertinggi,
                               "Final Location": lokasi}
            }
        }
        return Response(data=response, status=status.HTTP_200_OK)
    
    def check_different_values(self, data):
        # Jika tidak ada data, tidak ada nilai yang berbeda
        if not data:
            return False

        # Ambil nilai dari kolom pertama
        first_column_values = [item['username'] for item in data]

        # Jika ada nilai yang berbeda dalam data kolom pertama, return True
        if len(set(first_column_values)) > 1:
            return True

        return False
    


    # def list(self, request, *args, **kwargs):
    #     queryset = self.filter_queryset(self.get_queryset())
    #     serializer = self.get_serializer(queryset, many=True)
    #     data = serializer.data
    #     # queryset = self.filter_queryset(self.get_queryset())
    #     # serializer = self.get_serializer(queryset, many=True)
    #     # data = serializer.data

    #     if self.check_different_values(data):
            
    #         return Response(data=data, status=status.HTTP_200_OK)
        
        # # // Mengambil model dari database
        # data_name = data[0] if data else None
        # username = data_name.get('username') if data_name else None
        # model_with_pickle_instance = ModelWithPickle.objects.get(username=username)
        # svm_model = pickle.loads(model_with_pickle_instance.svm_model)
        # rf_model = pickle.loads(model_with_pickle_instance.rf_model)

        
        # def make_predictions(svm_model, rf_model, input_data):
        #     svm_prediction = svm_model.predict(input_data)
        #     rf_prediction = rf_model.predict(input_data)
            
        #     return svm_prediction, rf_prediction
        
        # data_background_service, background_data = PredictLocation(username)

        # # Make predictions
        # svm_prediction, rf_prediction = make_predictions(svm_model, rf_model, data_background_service)

    #     # Hitung kemunculan setiap string dalam array untuk kedua model
    #     counts_SVM = Counter(svm_prediction)
    #     counts_RF = Counter(rf_prediction)

    #     # Hitung total kemunculan untuk setiap model
    #     total_SVM = sum(counts_SVM.values())
    #     total_RF = sum(counts_RF.values())

    #     # Hitung dan cetak persentase setiap elemen untuk model SVM
    #     persentase_SVM = {}
    #     for nilai, jumlah in counts_SVM.items():
    #         persentase = (jumlah / total_SVM) * 100
    #         persentase_SVM[nilai] = "{:.2f} %".format(persentase)

    #     # Hitung dan cetak persentase setiap elemen untuk model RF
    #     persentase_RF = {}
    #     for nilai, jumlah in counts_RF.items():
    #         persentase = (jumlah / total_RF) * 100
    #         persentase_RF[nilai] = "{:.2f} %".format(persentase)

    #     # Dapatkan hasil prediksi dengan persentase terbesar
    #     max_svm_prediction = max(counts_SVM, key=counts_SVM.get)
    #     max_rf_prediction = max(counts_RF, key=counts_RF.get)
    #     # Hitung persentase untuk hasil prediksi terbesar
    #     persentase_max_svm_prediction = "{:.2f} %".format((counts_SVM[max_svm_prediction] / total_SVM) * 100) if total_SVM != 0 else 0
    #     persentase_max_rf_prediction = "{:.2f} %".format((counts_RF[max_rf_prediction] / total_RF) * 100) if total_RF != 0 else 0

    #     def insert_hasil_prediksi_to_database(username, lokasi, persentase_model_RF, persentase_model_SVM):
    #         # Buat instance serializer dengan data yang ingin dimasukkan
    #         data = {
    #             'username': username,
    #             'lokasi': lokasi,
    #             'persentase_model_Random_Forest': persentase_model_RF,
    #             'persentase_model_SVM': persentase_model_SVM
    #         }
    #         serializer = HasilPrediksiSerializer(data=data)

    #         # Periksa apakah data valid
    #         if serializer.is_valid():
    #             # Simpan data ke dalam database
    #             serializer.save()
    #             print("Data berhasil dimasukkan ke dalam database.")
    #         else:
    #             # Jika data tidak valid, cetak pesan kesalahan
    #             print("Gagal memasukkan data ke dalam database. Kesalahan:", serializer.errors)

    #     insert_hasil_prediksi_to_database(username, max_svm_prediction, persentase_max_rf_prediction, persentase_max_svm_prediction)

    #     # Buat respons
    #     response = {
    #         "Username": username,
    #         "SVM Percentage": persentase_SVM,
    #         "Max SVM Prediction": {"Value": max_svm_prediction, "Percentage": persentase_max_svm_prediction},
    #         "RF Percentage": persentase_RF,
    #         "Max RF Prediction": {"Value": max_rf_prediction, "Percentage": persentase_max_rf_prediction},
    #         # "Background Data": background_data
    #     }
    #     return Response(data=data, status=status.HTTP_200_OK)
    
    # def check_different_values(self, data):
    #     # Jika tidak ada data, tidak ada nilai yang berbeda
    #     if not data:
    #         return False

    #     # Ambil nilai dari kolom pertama
    #     first_column_values = [item['username'] for item in data]

    #     # Jika ada nilai yang berbeda dalam data kolom pertama, return True
    #     if len(set(first_column_values)) > 1:
    #         return True

    #     return False

    
