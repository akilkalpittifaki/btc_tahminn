import sys
import pytz
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QMessageBox)
from PyQt5.QtCore import Qt
import pytz
class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")
        self.setGeometry(100, 100, 800, 600)
        
        # Ana widget ve layout oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # Grafik için bir figure oluştur
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Tarih giriş alanı
        date_layout = QVBoxLayout()
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_input)
        layout.addLayout(date_layout)
        
        # Tahmin butonu
        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.clicked.connect(self.make_prediction)
        layout.addWidget(self.predict_button)
        
        # Veri ve model hazırlığı
        self.prepare_data_and_model()
        self.plot_bitcoin_data()

    def prepare_data_and_model(self):
        try:
            self.ticker = "BTC-USD"
            start_date = "2024-04-01"
            end_date = datetime.now().strftime("%Y-%m-%d")
            
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            
            # Hedef değişken oluştur
            self.df['Target'] = self.df['Close'].shift(-1)
            self.df.dropna(inplace=True)
            
            # Model özellikleri
            X = self.df[['Open', 'High', 'Low', 'Volume']]
            y = self.df['Target']
            
            # Ölçekleme
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Model eğitimi
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def plot_bitcoin_data(self):
        try:
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax)
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Grafik çizim hatası: {str(e)}")



    def predict_next_days(self, target_date_str):
        try:
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            # target_date'e zaman dilimi bilgisi ekleyin
            target_date = pytz.timezone('UTC').localize(target_date) 

            # Geçmiş tarih mi kontrol et
            if target_date <= self.df.index[-1]:
                # Tarihe kadar olan verileri filtrele
                df_past = self.df[self.df.index <= target_date]

                # Son günü tahmin için kullan
                last_data = df_past[['Open', 'High', 'Low', 'Volume']].iloc[-1].values
                scaled_data = self.scaler.transform(last_data)
                predicted_price = self.model.predict(scaled_data)[0]

                # Gerçek fiyat
                actual_price = df_past['Close'].iloc[-1]

                # Doğruluk oranını hesapla
                accuracy = 100 - (abs(predicted_price - actual_price) / actual_price) * 100

                # Sonuçları döndür (DataFrame yerine doğruluk oranı, gerçek fiyat ve tahmini fiyat)
                return accuracy, actual_price, predicted_price

            else:
                # Gelecek tarih ise, mevcut kodunuzu kullanın
                last_date = self.df.index[-1]
                days_to_predict = (target_date - last_date.to_pydatetime()).days

                predictions = []
                dates = []
                last_data = df_past[['Open', 'High', 'Low', 'Volume']].iloc[-1].values
                current_price = self.df['Close'].iloc[-1]

                for i in range(days_to_predict):
                    # Son veriyi ölçekle
                    scaled_data = self.scaler.transform(last_data)
                    
                    # Tahmin yap
                    next_price = self.model.predict(scaled_data)[0]
                    
                    # Tarihi ve tahmini kaydet
                    next_date = last_date + timedelta(days=i+1)
                    dates.append(next_date)
                    predictions.append(next_price)

                    # Bir sonraki tahmin için veriyi güncelle
                    last_data = np.array([[
                        next_price,  # Open
                        next_price * 1.02,  # High
                        next_price * 0.98,  # Low
                        last_data[0][3]  # Volume (aynı kalıyor)
                    ]])

                # Tahminleri DataFrame'e dönüştür
                predictions_df = pd.DataFrame({
                    'Tahmin': predictions
                }, index=dates)
                
                return predictions_df

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hesaplama hatası: {str(e)}")
            return None, None, None  # Hata durumunda None döndür

    def make_prediction(self):
        try:
            target_date_str = self.date_input.text()
            target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            target_date = pytz.timezone('UTC').localize(target_date)  # Zaman dilimi bilgisi ekleyin

            result = self.predict_next_days(target_date_str)

            if result[0] is None:  # Hata durumunu kontrol et
                return

            if isinstance(result[0], float):  # Geçmiş tarih için sonuç
                accuracy, actual_price, predicted_price = result

                # Tahmin nasıl yapıldı açıklaması
                explanation = f"Model, {target_date_str} tarihinden önceki Bitcoin fiyatlarını (açılış, yüksek, düşük ve işlem hacmi) inceleyerek bir tahmin oluşturdu.\n"
                explanation += "Bu veriler, geçmişteki fiyat hareketlerini öğrenen bir makine öğrenimi modeli tarafından analiz edildi.\n"
                explanation += f"Model, {target_date_str} tarihi için {predicted_price:.2f} dolarlık bir fiyat tahmin etti.\n"
                explanation += f"Gerçekleşen fiyat ise {actual_price:.2f} dolardı.\n"
                explanation += f"Bu da %{accuracy:.2f} oranında bir doğruluğa denk gelmektedir."

                result_text = f"Geçmiş tarih için tahmin sonuçları:\n\n"
                result_text += f"Tarih: {target_date_str}\n"
                result_text += f"Gerçek Fiyat: ${actual_price:.2f}\n"
                result_text += f"Tahmini Fiyat: ${predicted_price:.2f}\n"
                result_text += f"Doğruluk Oranı: %{accuracy:.2f}\n\n"
                result_text += explanation  # Açıklamayı ekle

                QMessageBox.information(self, "Tahmin Sonuçları", result_text)

            else:  # Gelecek tarih için sonuç
                predictions_df = result
                # Grafiği güncelle
                self.ax.clear()
                self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler')
                predictions_df['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--')
                self.ax.legend()
                self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
                self.ax.grid(True)
                self.canvas.draw()
                
                # Sonuçları göster
                result_text = "Tahmin Sonuçları:\n\n"
                for idx, row in predictions_df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    prediction_str = f"{row['Tahmin']:.2f}"
                    result_text += f"Tarih: {date_str}\n"
                    result_text += f"Tahmin: ${prediction_str}\n\n"
                
                QMessageBox.information(self, "Tahmin Sonuçları", result_text)
                
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())