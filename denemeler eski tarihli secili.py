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

# Ana sınıf olan BitcoinPredictorApp tanımlanıyor
class BitcoinPredictorApp(QMainWindow):  
    def __init__(self):  
        super().__init__()  
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")  # Uygulama başlığını ayarla
        self.setGeometry(100, 100, 800, 600)  # Pencere boyutunu ayarla
        
        # Ana widget ve layout oluştur
        main_widget = QWidget()  
        self.setCentralWidget(main_widget)  
        layout = QVBoxLayout(main_widget)  
        
        # Grafik için bir figure (şekil) oluştur ve arayüze ekle
        self.figure, self.ax = plt.subplots(figsize=(8, 6))  
        self.canvas = FigureCanvas(self.figure)  # Matplotlib çizimini PyQt5 ile görüntülemek için canvas kullanılır
        layout.addWidget(self.canvas)  
        
        # Tahmin yapılacak tarih için giriş alanı ekle
        date_layout = QVBoxLayout()  
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")  # Tarih giriş etiketi
        self.date_input = QLineEdit()  # Tarih giriş kutusu
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")  # Örnek tarih formatı için açıklama
        date_layout.addWidget(self.date_label)  
        date_layout.addWidget(self.date_input)  
        layout.addLayout(date_layout)  
        
        # Tahmin etme işlemi için bir buton ekle
        self.predict_button = QPushButton("Tahmin Et")  
        self.predict_button.clicked.connect(self.make_prediction)  # Butona tıklama işlemi bağlantısı
        layout.addWidget(self.predict_button)  
        
        # Veri ve model hazırlığı fonksiyonunu çağır
        self.prepare_data_and_model()  
        # Bitcoin verisini çiz
        self.plot_bitcoin_data()  

    def prepare_data_and_model(self):  
        """Veri setini hazırla ve modeli eğit"""
        # Bitcoin fiyat verilerini Yahoo Finance üzerinden al  
        self.ticker = "BTC-USD"  # Bitcoin-USD sembolü
        start_date = "2024-04-01"  # Veri başlangıç tarihi
        end_date = datetime.now().strftime("%Y-%m-%d")  # Veri bitiş tarihi (bugünkü tarih)
        
        try:  
            # Veriyi indir
            self.df = yf.download(self.ticker, start=start_date, end=end_date)  
            
            # Hedef değişken oluştur: Bir sonraki kapanış fiyatı için
            self.df.loc[:, 'preds'] = self.df['Close'].shift(-1)  # 'Close' fiyatını bir sonraki güne kaydırarak hedef değişken olarak kullan
            self.df.dropna(inplace=True)  # Boş verileri çıkar
            
            # Model için özellikleri hazırla
            X = self.df[['Open', 'High', 'Low', 'Volume']]  # Girdi değişkenleri (özellikler)
            y = self.df['preds']  # Hedef değişken
            
            # Özellikleri ölçekle
            self.scaler = StandardScaler()  # Verileri normalize etmek için kullanılır
            X_scaled = self.scaler.fit_transform(X)  # Özellikleri ölçekle
            
            # Modeli oluştur ve eğit
            self.model = LinearRegression()  # Lineer regresyon modeli oluştur
            self.model.fit(X_scaled, y)  # Modeli eğit
            
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")  # Hata durumunda mesaj kutusu göster

    def plot_bitcoin_data(self):  
        """Bitcoin verilerini grafiğe çiz"""  
        try:  
            # Grafik temizleme ve Bitcoin verilerini çizme
            self.ax.clear()  
            self.df['Close'].plot(ax=self.ax)  # Kapanış fiyatlarını grafiğe ekle
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')  # Grafik başlığı
            self.ax.set_xlabel('Tarih')  # X ekseni etiketi
            self.ax.set_ylabel('Fiyat (USD)')  # Y ekseni etiketi
            self.ax.grid(True)  # Grafik ızgarasını göster
            self.canvas.draw()  # Grafiği güncelle
        except Exception as e:  
            QMessageBox.warning(self, "Uyarı", f"Grafik çizim hatası: {str(e)}")  # Hata mesajı göster

    def predict_next_days(self, end_date):  
        """Gelecek günler için tahmin yap"""  
        last_real_date = self.df.index[-1].tz_localize(None)  # Son gerçek veri tarihi
        target_date = datetime.strptime(end_date, '%Y-%m-%d')  # Hedef tarih
        days_to_predict = (target_date - last_real_date).days  # Tahmin yapılacak gün sayısı
        
        if days_to_predict <= 0:  
            # Eğer hedef tarih geçmişte ise, o tarihten önceki 6 aylık verileri kullanarak tahmin yap
            start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
            try:
                df_past = yf.download(self.ticker, start=start_date, end=target_date.strftime('%Y-%m-%d'))
                df_past.loc[:, 'preds'] = df_past['Close'].shift(-1)
                df_past.dropna(inplace=True)
                
                X_past = df_past[['Open', 'High', 'Low', 'Volume']]
                y_past = df_past['preds']
                X_past_scaled = self.scaler.fit_transform(X_past)
                
                # Mevcut model ile geçmiş veriler üzerinde tahmin yap
                y_pred = self.model.predict(X_past_scaled)
                accuracy = 1 - np.mean(np.abs((y_past - y_pred) / y_past))  # Ortalama mutlak yüzde hata (MAPE) kullanarak doğruluk hesapla
                
                QMessageBox.information(self, "Tahmin Sonucu", f"Geçmiş tarih için tahmin yapıldı. Doğruluk oranı: {accuracy:.2%}")
                return pd.DataFrame()
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Geçmiş veri tahmini hatası: {str(e)}")
                return pd.DataFrame()
        
        predictions = []  # Tahminleri saklamak için liste
        dates = []  # Tahmin tarihlerini saklamak için liste
        current_features = self.df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values  # Son özellik değerleri
        
        for i in range(days_to_predict):  
            current_scaled = self.scaler.transform(current_features)  # Özellikleri ölçekle
            next_price = self.model.predict(current_scaled)[0]  # Modelden fiyat tahmini
            
            next_date = last_real_date + timedelta(days=i+1)  # Sonraki günün tarihini hesapla
            predictions.append(next_price)  # Tahmin fiyatını ekle
            dates.append(next_date)  # Tarihi ekle
            
            # Sonraki özellikleri güncelle
            current_features = np.array([[  
                float(next_price),  
                float(next_price * 1.01),  # High tahmini %1 fazlası olarak belirleniyor
                float(next_price * 0.99),  # Low tahmini %1 eksik olarak belirleniyor
                float(current_features[0][3])  # Hacim aynı tutuluyor
            ]])  
        
        return pd.DataFrame({  
            'Tarih': dates,  
            'Tahmin': predictions  
        })  

    def make_prediction(self):  
        """Tahmin butonuna tıklandığında çalışacak fonksiyon"""  
        try:  
            end_date = self.date_input.text()  # Girdi tarihini al
            datetime.strptime(end_date, '%Y-%m-%d')  # Tarih formatını kontrol et
            
            predictions_df = self.predict_next_days(end_date)  # Tahmin yap
            
            if predictions_df.empty:  
                return  
            
            # Tahminleri grafiğe ekle
            self.ax.clear()  
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler')  # Gerçek verileri çiz
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--')  # Tahminleri çiz
            self.ax.legend()  # Grafik için açıklama
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')  # Grafik başlığı
            self.ax.grid(True)  
            self.canvas.draw()  
            
            # Tahmin sonuçlarını mesaj kutusunda göster
            result_text = "Tahmin Sonuçları:\n\n"  
            for _, row in predictions_df.iterrows():  
                result_text += f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"  
                result_text += f"Tahmin: ${row['Tahmin']:.2f}\n\n"  
            
            QMessageBox.information(self, "Tahmin Sonuçları", result_text)  
            
        except ValueError:  
            QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")  
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")  # Hata mesajı göster

# Uygulama çalıştırma kodu
if __name__ == '__main__':  
    app = QApplication(sys.argv)  # PyQt5 uygulaması başlatılıyor
    window = BitcoinPredictorApp()  # Uygulama penceresini oluştur
    window.show()  # Pencereyi göster
    sys.exit(app.exec_())  # Uygulamayı çalıştır
