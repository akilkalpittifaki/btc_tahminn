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
                           QPushButton, QLabel, QLineEdit, QMessageBox,
                           QTextEdit, QSplitter)  # QTextEdit ve QSplitter eklendi
from PyQt5.QtCore import Qt

class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")
        self.setGeometry(100, 100, 1000, 800)  # Pencere boyutu artırıldı
        
        # Ana widget ve splitter oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Splitter oluştur - Grafik ve sonuçları yan yana gösterecek
        splitter = QSplitter(Qt.Horizontal)
        
        # Sol taraf - Grafik ve giriş alanları
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Grafik için figure oluştur
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)
        
        # Tarih girişi için alan
        date_widget = QWidget()
        date_layout = QVBoxLayout(date_widget)
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_input)
        
        # Tahmin butonu
        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.clicked.connect(self.make_prediction)
        date_layout.addWidget(self.predict_button)
        
        left_layout.addWidget(date_widget)
        
        # Sağ taraf - Sonuçlar için metin alanı
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(300)  # Minimum genişlik ayarla
        
        # Splitter'a widget'ları ekle
        splitter.addWidget(left_widget)
        splitter.addWidget(self.results_text)
        
        # Ana layout'a splitter'ı ekle
        main_layout.addWidget(splitter)
        
        # Veri ve model hazırlığı
        self.prepare_data_and_model()
        self.plot_bitcoin_data()

    def prepare_data_and_model(self):
        # ... (mevcut kod aynı kalacak)
        """Veri setini hazırla ve modeli eğit"""
        self.ticker = "BTC-USD"
        start_date = "2024-04-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            self.df.loc[:, 'preds'] = self.df['Close'].shift(-1)
            self.df.dropna(inplace=True)
            
            X = self.df[['Open', 'High', 'Low', 'Volume']]
            y = self.df['preds']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def plot_bitcoin_data(self):
        # ... (mevcut kod aynı kalacak)
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
        # ... (mevcut kod aynı kalacak)
        last_real_date = self.df.index[-1].tz_localize(None)
        target_date = datetime.strptime(end_date, '%Y-%m-%d')
        days_to_predict = (target_date - last_real_date).days

        if days_to_predict <= 0:
            start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
            try:
                df_past = yf.download(self.ticker, start=start_date, end=target_date.strftime('%Y-%m-%d'))
                df_past.loc[:, 'preds'] = df_past['Close'].shift(-1)
                df_past.dropna(inplace=True)
                
                X_past = df_past[['Open', 'High', 'Low', 'Volume']]
                y_past = df_past['preds']
                X_past_scaled = self.scaler.fit_transform(X_past)
                
                y_pred = self.model.predict(X_past_scaled)
                accuracy = 1 - np.mean(np.abs((y_past - y_pred) / y_past))
                
                QMessageBox.information(self, "Tahmin Sonucu", f"Geçmiş tarih için tahmin yapıldı. Doğruluk oranı: {accuracy:.2%}")
                return pd.DataFrame()
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Geçmiş veri tahmini hatası: {str(e)}")
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
        try:
            end_date = self.date_input.text()
            datetime.strptime(end_date, '%Y-%m-%d')
            
            predictions_df = self.predict_next_days(end_date)
            
            if predictions_df.empty:
                return
            
            # Grafiği güncelle
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler')
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--')
            self.ax.legend()
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.grid(True)
            self.canvas.draw()
            
            # Sonuçları metin alanına yaz
            self.results_text.clear()
            self.results_text.append("Tahmin Sonuçları:\n")
            for _, row in predictions_df.iterrows():
                self.results_text.append(
                    f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"
                    f"Tahmin: ${row['Tahmin']:.2f}\n\n"
                )
            
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())