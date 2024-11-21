import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QMessageBox,
                           QTextEdit, QSplitter, QDialog)
from PyQt5.QtCore import Qt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from fbprophet import Prophet  # Prophet kütüphanesini ekledik


class AnalysisDialog(QDialog):
    def __init__(self, analysis_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tahmin Analizi")
        self.setGeometry(200, 200, 800, 600)
        
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setPlainText(analysis_text)
        
        layout.addWidget(text_edit)
        
        close_button = QPushButton("Kapat")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)
        
        self.setLayout(layout)

class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")
        self.setGeometry(100, 100, 1200, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)
        
        date_widget = QWidget()
        date_layout = QVBoxLayout(date_widget)
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_input)
        
        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.clicked.connect(self.make_prediction)
        date_layout.addWidget(self.predict_button)
        
        left_layout.addWidget(date_widget)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(300)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.results_text)
        
        main_layout.addWidget(splitter)
        
        self.prepare_data_and_models()
        self.plot_bitcoin_data()

    def prepare_data_and_models(self):
        """Veri hazırlama ve model eğitimi"""
        self.ticker = "BTC-USD"
        # 4 yıllık veri al
        start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            # Verileri indir
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            
            # Prophet için veri hazırlığı
            self.prepare_prophet_data()
            
            # LSTM için veri hazırlığı
            self.prepare_lstm_sequences()
            
            # Modelleri eğit
            self.train_prophet_model()
            self.train_lstm_model()
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def prepare_prophet_data(self):
        """Prophet modeli için veriyi hazırla"""
        self.prophet_df = self.df.reset_index()[['Date', 'Close']]
        self.prophet_df['ds'] = pd.to_datetime(self.prophet_df['ds']).dt.tz_localize(None)
        self.prophet_df.columns = ['ds', 'y']  # Prophet için gerekli kolon isimleri

    def train_prophet_model(self):
        """Prophet modelini eğit"""
        self.prophet_model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        self.prophet_model.fit(self.prophet_df)

    def prepare_lstm_sequences(self, time_step=60):
        """LSTM için çok değişkenli zaman serisi verisi hazırla"""
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume']
        self.df_features = self.df[self.features]
        
        # MinMaxScaler kullanarak veriyi ölçeklendir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler.fit_transform(self.df_features)
        
        self.X_lstm = []
        self.y_lstm = []
        
        # Tüm özellikler için sekanslar oluştur
        for i in range(time_step, len(self.X_scaled)):
            self.X_lstm.append(self.X_scaled[i-time_step:i])
            # Sadece kapanış fiyatını hedef olarak al
            self.y_lstm.append(self.X_scaled[i, 3])  # 3. indeks 'Close' fiyatını temsil ediyor
        
        self.X_lstm = np.array(self.X_lstm)
        self.y_lstm = np.array(self.y_lstm)
        
        print(f"X_lstm shape: {self.X_lstm.shape}")
        print(f"y_lstm shape: {self.y_lstm.shape}")

    def train_lstm_model(self):
        """LSTM modelini eğit"""
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(self.X_lstm.shape[1], self.X_lstm.shape[2])))
        self.lstm_model.add(LSTM(units=50, return_sequences=False))
        self.lstm_model.add(Dense(units=1))

        self.lstm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        self.lstm_model.fit(self.X_lstm, self.y_lstm, epochs=50, batch_size=32, validation_split=0.2)

    def plot_bitcoin_data(self):
        """Bitcoin verilerini göster"""
        try:
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Bitcoin Fiyatı', linewidth=2)
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Grafik çizim hatası: {str(e)}")

    def make_prediction(self):
        """Tahmin yap"""
        try:
            end_date = self.date_input.text()
            datetime.strptime(end_date, '%Y-%m-%d')
            
            predictions_df = self.predict_with_prophet(end_date)
            
            if predictions_df.empty:
                return
            
            # Grafiği güncelle
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2)
            predictions_df.set_index('ds')['yhat'].plot(ax=self.ax, 
                                                        label='Tahminler',
                                                        style='--',
                                                        linewidth=2)
            self.ax.legend()
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.grid(True)
            self.canvas.draw()
            
            # Sonuçları göster
            self.results_text.clear()
            self.results_text.append("Tahmin Sonuçları:\n")
            for _, row in predictions_df.iterrows():
                self.results_text.append(
                    f"Tarih: {row['ds'].strftime('%Y-%m-%d')}\n"
                    f"Tahmin: ${row['yhat']:.2f}\n\n"
                )
        
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")

    def predict_with_prophet(self, end_date):
        """Prophet modeli ile geleceği tahmin et"""
        try:
            future = self.prophet_model.make_future_dataframe(periods=365)
            forecast = self.prophet_model.predict(future)
            forecast = forecast[forecast['ds'] <= end_date]
            return forecast[['ds', 'yhat']]
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Prophet tahmini hatası: {str(e)}")
            return pd.DataFrame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
