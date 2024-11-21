import sys  
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

class BitcoinPredictorApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")  
        self.setGeometry(100, 100, 800, 600)  
        
        # Ana widget ve layout  
        main_widget = QWidget()  
        self.setCentralWidget(main_widget)  
        layout = QVBoxLayout(main_widget)  
        
        # Grafik için figure oluştur  
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
        """Veri setini hazırla ve modeli eğit"""  
        # Bitcoin verilerini çek  
        self.ticker = "BTC-USD"  
        start_date = "2024-04-01"  
        end_date = datetime.now().strftime("%Y-%m-%d")  
        
        try:  
            self.df = yf.download(self.ticker, start=start_date, end=end_date)  
            
            # Hedef değişkeni oluştur  
            self.df.loc[:, 'preds'] = self.df['Close'].shift(-1)  
            self.df.dropna(inplace=True)  
            
            # Model için verileri hazırla  
            X = self.df[['Open', 'High', 'Low', 'Volume']]  
            y = self.df['preds']  
            
            # Veriyi ölçekle  
            self.scaler = StandardScaler()  
            X_scaled = self.scaler.fit_transform(X)  
            
            # Modeli eğit  
            self.model = LinearRegression()  
            self.model.fit(X_scaled, y)  
            
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")  

    def plot_bitcoin_data(self):  
        """Bitcoin verilerini grafiğe çiz"""  
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

    def predict_next_days(self, end_date):  
        """Gelecek günler için tahmin yap"""  
        last_real_date = self.df.index[-1].tz_localize(None)  
        target_date = datetime.strptime(end_date, '%Y-%m-%d')  
        days_to_predict = (target_date - last_real_date).days  
        
        if days_to_predict <= 0:  
            return pd.DataFrame()  
        
        predictions = []  
        dates = []  
        current_features = self.df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values  
        
        for i in range(days_to_predict):  
            current_scaled = self.scaler.transform(current_features)  
            next_price = self.model.predict(current_scaled)[0]  
            
            next_date = last_real_date + timedelta(days=i+1)  
            predictions.append(next_price)  
            dates.append(next_date)  
            
            current_features = np.array([[  
                float(next_price),  
                float(next_price * 1.01),  
                float(next_price * 0.99),  
                float(current_features[0][3])  
            ]])  
        
        return pd.DataFrame({  
            'Tarih': dates,  
            'Tahmin': predictions  
        })  

    def make_prediction(self):  
        """Tahmin butonuna tıklandığında çalışacak fonksiyon"""  
        try:  
            end_date = self.date_input.text()  
            datetime.strptime(end_date, '%Y-%m-%d')  # Tarih formatını kontrol et  
            
            predictions_df = self.predict_next_days(end_date)  
            
            if predictions_df.empty:  
                QMessageBox.warning(self, "Uyarı", "Geçerli bir tarih giriniz!")  
                return  
            
            # Tahminleri grafiğe ekle  
            self.ax.clear()  
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler')  
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--')  
            self.ax.legend()  
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')  
            self.ax.grid(True)  
            self.canvas.draw()  
            
            # Tahmin sonuçlarını göster  
            result_text = "Tahmin Sonuçları:\n\n"  
            for _, row in predictions_df.iterrows():  
                result_text += f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"  
                result_text += f"Tahmin: ${row['Tahmin']:.2f}\n\n"  
            
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