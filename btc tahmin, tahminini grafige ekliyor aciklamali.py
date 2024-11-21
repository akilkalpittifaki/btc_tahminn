# Gerekli kütüphaneler import ediliyor
import sys  # Python sistem çağrıları ve çıkış işleri için kullanılır
import pandas as pd  # Veri işlemleri ve analizleri için Pandas kullanılıyor
import yfinance as yf  # Yahoo Finance'den finansal veri çekmek için kullanılır
import numpy as np  # Sayısal hesaplamalar ve matris işlemleri için Numpy kullanılıyor
from datetime import datetime, timedelta  # Tarih ve zaman işlemleri için kullanılır
from sklearn.preprocessing import StandardScaler  # Veriyi ölçeklemek için kullanılır
from sklearn.linear_model import LinearRegression  # Lineer regresyon modeli oluşturmak için kullanılır
import matplotlib.pyplot as plt  # Grafik çizimleri için Matplotlib kullanılıyor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas  # Matplotlib'i PyQt5 ile kullanmak için
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,   
                           QPushButton, QLabel, QLineEdit, QMessageBox)  # PyQt5 arayüz öğeleri için kullanılır
from PyQt5.QtCore import Qt  # Qt çekirdek modülü, pencere ve widget işlemleri için kullanılır

"""  
YAPAY ZEKA VE TAHMİN ALGORİTMASI AÇIKLAMASI:  

1. Kullanılan Yapay Zeka Modeli: Linear Regression (Lineer Regresyon)  
   - Bu model, birden fazla girdi değişkeni (features) kullanarak çıktı değişkenini (target) tahmin eder  
   - Matematiksel olarak: y = b0 + b1x1 + b2x2 + b3x3 + b4x4 formülünü kullanır  
   - Burada:  
     * y: Tahmin edilen Bitcoin fiyatı  
     * b0: Sabit terim (bias)  
     * b1,b2,b3,b4: Her bir özellik için ağırlık katsayıları  
     * x1,x2,x3,x4: Girdi özellikleri (Open, High, Low, Volume)  

2. Tahmin Süreci:  
   a) Veri Hazırlama:  
      - Geçmiş Bitcoin verileri (Open, High, Low, Close, Volume) toplanır  
      - Veriler normalize edilir (StandardScaler ile)  
      - Hedef değişken: Bir sonraki günün kapanış fiyatı  

   b) Model Eğitimi:  
      - Model, geçmiş verilerdeki pattern'leri öğrenir  
      - Özellikler ile hedef arasındaki ilişkiyi matematiksel olarak modeller  

   c) Tahmin:  
      - Son gün verileri kullanılarak gelecek tahmin edilir  
      - Her tahmin bir sonraki gün için girdi olarak kullanılır  

3. Tahmin Dayanak Noktaları:  
   - Açılış Fiyatı (Open): Günün başlangıç fiyatı  
   - En Yüksek Fiyat (High): Gün içi en yüksek değer  
   - En Düşük Fiyat (Low): Gün içi en düşük değer  
   - İşlem Hacmi (Volume): Alım-satım miktarı  

4. Modelin Güçlü ve Zayıf Yönleri:  
   Güçlü Yönler:  
   - Basit ve anlaşılır matematik modeli  
   - Hızlı eğitim ve tahmin süreci  
   - Az kaynak kullanımı  

   Zayıf Yönler:  
   - Karmaşık piyasa dinamiklerini tam yakalayamama  
   - Sadece doğrusal ilişkileri modelleyebilme  
   - Ani piyasa değişimlerini öngörememe  
"""  

class BitcoinPredictorApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")  
        self.setGeometry(100, 100, 800, 600)  
        
        # GUI bileşenleri oluşturuluyor  
        main_widget = QWidget()  
        self.setCentralWidget(main_widget)  
        layout = QVBoxLayout(main_widget)  
        
        self.figure, self.ax = plt.subplots(figsize=(8, 6))  
        self.canvas = FigureCanvas(self.figure)  
        layout.addWidget(self.canvas)  
        
        date_layout = QVBoxLayout()  
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")  
        self.date_input = QLineEdit()  
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")  
        date_layout.addWidget(self.date_label)  
        date_layout.addWidget(self.date_input)  
        layout.addLayout(date_layout)  
        
        self.predict_button = QPushButton("Tahmin Et")  
        self.predict_button.clicked.connect(self.make_prediction)  
        layout.addWidget(self.predict_button)  
        
        self.prepare_data_and_model()  
        self.plot_bitcoin_data()  

    def prepare_data_and_model(self):  
        """  
        YAPAY ZEKA MODEL HAZIRLAMA VE EĞİTME SÜRECİ  
        
        1. Veri Toplama:  
           - Yahoo Finance API'den Bitcoin verileri çekilir  
           - Tarih aralığı: 2024-04-01'den bugüne  
        
        2. Veri Hazırlama:  
           - Features (Özellikler): [Open, High, Low, Volume]  
           - Target (Hedef): Bir sonraki günün kapanış fiyatı  
           - StandardScaler ile veriler normalize edilir (0 ortalama, 1 standart sapma)  
        
        3. Model Eğitimi:  
           - LinearRegression modeli kullanılır  
           - Model, geçmiş verilerdeki ilişkileri öğrenir  
           - Matematiksel formül: next_price = b0 + b1*open + b2*high + b3*low + b4*volume  
        """  
        try:  
            self.ticker = "BTC-USD"  
            start_date = "2024-04-01"  
            end_date = datetime.now().strftime("%Y-%m-%d")  
            
            # Veri çekme ve hazırlama  
            self.df = yf.download(self.ticker, start=start_date, end=end_date)  
            
            # Hedef değişken oluşturma (bir sonraki günün kapanış fiyatı)  
            self.df.loc[:, 'preds'] = self.df['Close'].shift(-1)  
            self.df.dropna(inplace=True)  
            
            # Model özellikleri ve hedef  
            X = self.df[['Open', 'High', 'Low', 'Volume']]  
            y = self.df['preds']  
            
            # Veri normalizasyonu  
            self.scaler = StandardScaler()  
            X_scaled = self.scaler.fit_transform(X)  
            
            # Model eğitimi  
            self.model = LinearRegression()  
            self.model.fit(X_scaled, y)  
            
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")  

    def predict_next_days(self, end_date):  
        """  
        TAHMİN YAPMA ALGORİTMASI  
        
        1. Tahmin Süreci:  
           a) Son günün verileri alınır  
           b) Veriler normalize edilir  
           c) Model ile tahmin yapılır  
           d) Tahmin sonuçları bir sonraki gün için girdi olarak kullanılır  
        
        2. Tahmin Parametreleri:  
           - Açılış Fiyatı: Bir önceki tahmin değeri  
           - En Yüksek: Tahmin + %1  
           - En Düşük: Tahmin - %1  
           - Hacim: Son gerçek hacim değeri (sabit tutulur)  
        
        3. Güven Aralığı:  
           - Tahminler %1'lik bir band içinde hareket eder  
           - Bu band, kripto para piyasasının volatilitesini temsil eder  
        """  
        last_real_date = self.df.index[-1].tz_localize(None)  
        target_date = datetime.strptime(end_date, '%Y-%m-%d')  
        days_to_predict = (target_date - last_real_date).days  
        
        if days_to_predict <= 0:  
            return pd.DataFrame()  
        
        predictions = []  
        dates = []  
        current_features = self.df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values  
        
        for i in range(days_to_predict):  
            # Veriyi normalize et ve tahmin yap  
            current_scaled = self.scaler.transform(current_features)  
            next_price = self.model.predict(current_scaled)[0]  
            
            next_date = last_real_date + timedelta(days=i+1)  
            predictions.append(next_price)  
            dates.append(next_date)  
            
            # Sonraki gün için özellikleri güncelle  
            current_features = np.array([[  
                float(next_price),  # Açılış = Son tahmin  
                float(next_price * 1.01),  # En yüksek = Tahmin + %1  
                float(next_price * 0.99),  # En düşük = Tahmin - %1  
                float(current_features[0][3])  # Hacim sabit  
            ]])  
        
        return pd.DataFrame({  
            'Tarih': dates,  
            'Tahmin': predictions  
        })  

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

    def make_prediction(self):  
        """  
        TAHMİN SONUÇLARINI GÖRSELLEŞTIRME VE GÖSTERME  
        
        1. Grafik Gösterimi:  
           - Mavi çizgi: Gerçek geçmiş veriler  
           - Turuncu kesikli çizgi: Gelecek tahminleri  
        
        2. Sayısal Sonuçlar:  
           - Her gün için tarih ve tahmini fiyat  
           - Sonuçlar bir mesaj kutusu içinde gösterilir  
        """  
        try:  
            end_date = self.date_input.text()  
            datetime.strptime(end_date, '%Y-%m-%d')  
            
            predictions_df = self.predict_next_days(end_date)  
            
            if predictions_df.empty:  
                QMessageBox.warning(self, "Uyarı", "Geçerli bir tarih giriniz!")  
                return  
            
            self.ax.clear()  
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler')  
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--')  
            self.ax.legend()  
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')  
            self.ax.grid(True)  
            self.canvas.draw()  
            
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