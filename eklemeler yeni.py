import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QMessageBox,
                           QTextEdit, QSplitter, QDialog)
from PyQt5.QtCore import Qt
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


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
            end_date = datetime.now().strftime('%Y-%m-%d')  # Buradaki hata düzeltildi
            
            try:
                # Verileri indir
                self.df = yf.download(self.ticker, start=start_date, end=end_date)
                
                # Teknik indikatörler ekle
                self.add_technical_indicators()
                
                # Veriyi eğitim için hazırla
                self.prepare_training_data()
                
                # Modelleri eğit
                self.train_models()
                
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def add_technical_indicators(self):
        """Teknik indikatörleri hesapla"""
        try:
            # Returns ve volatilite
            self.df['Returns'] = self.df['Close'].pct_change()
            self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
            
            # Hareketli ortalamalar
            self.df['MA7'] = self.df['Close'].rolling(window=7).mean()
            self.df['MA30'] = self.df['Close'].rolling(window=30).mean()
            self.df['MA90'] = self.df['Close'].rolling(window=90).mean()
            self.df['MA200'] = self.df['Close'].rolling(window=200).mean()
            
            # RSI
            self.df['RSI'] = self.calculate_rsi(self.df['Close'])
            
            # MACD
            exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
            self.df['MACD'] = exp1 - exp2
            self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
            
            # Bollinger Bands düzeltilmiş hesaplama
            bb_window = 20
            rolling_mean = self.df['Close'].rolling(window=bb_window).mean()
            rolling_std = self.df['Close'].rolling(window=bb_window).std()
            
            self.df['BB_middle'] = rolling_mean
            self.df['BB_upper'] = rolling_mean + (rolling_std * 2)
            self.df['BB_lower'] = rolling_mean - (rolling_std * 2)
            
            # Eksik değerleri temizle
            self.df.dropna(inplace=True)
            
        except Exception as e:
            print(f"Teknik indikatör hesaplama hatası: {str(e)}")
            raise e

    def calculate_rsi(self, prices, period=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        # Linear Regression için özellikler
        self.linear_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 'MA7', 'MA30']
        
        # XGBoost için genişletilmiş özellikler
        self.xgb_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 
                            'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                            'RSI', 'MACD', 'Signal_Line', 
                            'BB_middle', 'BB_upper', 'BB_lower']
        
        # Scaler'ları oluştur
        self.linear_scaler = StandardScaler()
        self.xgb_scaler = StandardScaler()
        
        # Verileri ölçekle
        self.X_linear = self.df[self.linear_features]
        self.X_xgb = self.df[self.xgb_features]
        self.y = self.df['Close']
        
        self.X_linear_scaled = self.linear_scaler.fit_transform(self.X_linear)
        self.X_xgb_scaled = self.xgb_scaler.fit_transform(self.X_xgb)

    def train_models(self):
        """Modelleri eğit"""
        # Linear Regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(self.X_linear_scaled, self.y)
        
        # XGBoost
        self.xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            min_child_weight=2,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
        )
        self.xgb_model.fit(self.X_xgb_scaled, self.y)

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
            
            predictions_df = self.predict_next_days(end_date)
            
            if predictions_df.empty:
                return
            
            # Grafiği güncelle
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2)
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, 
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
                    f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"
                    f"Tahmin: ${row['Tahmin']:.2f}\n\n"
                )
            
        except ValueError:
            QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")

    def predict_next_days(self, end_date):
        """Gelecek günleri tahmin et"""
        last_real_date = self.df.index[-1].tz_localize(None)
        target_date = datetime.strptime(end_date, '%Y-%m-%d')
        days_to_predict = (target_date - last_real_date).days

        if days_to_predict <= 0:
            return self.analyze_past_date(target_date)
        else:
            return self.predict_future_date(days_to_predict, last_real_date)

    def predict_future_date(self, days_to_predict, last_real_date):
        """Gelecek tarihi tahmin et"""
        try:
            predictions = []
            dates = []
            model_weights = []
            current_data = self.df.iloc[-1:].copy()
            last_price = float(self.df['Close'].iloc[-1])
            
            for i in range(days_to_predict):
                # Modellerin ağırlıklarını belirle
                if i <= 7:
                    linear_weight = 0.7  # İlk 7 gün için Linear ağırlıklı
                    xgb_weight = 0.3
                else:
                    linear_weight = 0.3  # Sonraki günler için XGBoost ağırlıklı
                    xgb_weight = 0.7
                    
                # Linear Regression tahmini
                linear_features = current_data[self.linear_features]
                linear_scaled = self.linear_scaler.transform(linear_features)
                linear_pred = float(self.linear_model.predict(linear_scaled)[0])
                
                # XGBoost tahmini
                xgb_features = current_data[self.xgb_features]
                xgb_scaled = self.xgb_scaler.transform(xgb_features)
                xgb_pred = float(self.xgb_model.predict(xgb_scaled)[0])
                
                # Ağırlıklı tahmin
                next_price = (linear_pred * linear_weight) + (xgb_pred * xgb_weight)
                next_date = last_real_date + timedelta(days=i+1)
                
                predictions.append(next_price)
                dates.append(next_date)
                model_weights.append({
                    'Linear': linear_weight,
                    'XGBoost': xgb_weight,
                    'Linear_Pred': linear_pred,
                    'XGBoost_Pred': xgb_pred
                })
                
                # Bir sonraki gün için özellikleri güncelle
                self.update_technical_features(current_data, next_price, predictions)
                
            # Sonuçları DataFrame'e dönüştür
            future_df = pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions
            })
            
            # Detaylı analiz metni oluştur
            analysis_text = f"""Bitcoin Fiyat Tahmin Analizi

Tahmin Özeti:
1. Başlangıç Tarihi: {dates[0].strftime('%Y-%m-%d')}
2. Bitiş Tarihi: {dates[-1].strftime('%Y-%m-%d')}
3. Mevcut Fiyat: ${last_price:.2f}
4. Son Tahmin: ${predictions[-1]:.2f}
5. Toplam Beklenen Değişim: %{((predictions[-1] - last_price) / last_price * 100):.2f}

Günlük Tahmin Detayları:
{'='*50}"""

            for i, (date, pred, weight) in enumerate(zip(dates, predictions, model_weights)):
                daily_change = ((pred - (predictions[i-1] if i > 0 else last_price)) / 
                              (predictions[i-1] if i > 0 else last_price) * 100)
                total_change = ((pred - last_price) / last_price * 100)
                
                analysis_text += f"""
Tarih: {date.strftime('%Y-%m-%d')}
Tahmini Fiyat: ${pred:.2f}
Günlük Değişim: %{daily_change:.2f}
Toplam Değişim: %{total_change:.2f}
Model Ağırlıkları:
- Linear Regression (%{weight['Linear']*100:.0f}): ${weight['Linear_Pred']:.2f}
- XGBoost (%{weight['XGBoost']*100:.0f}): ${weight['XGBoost_Pred']:.2f}
{'='*50}"""

            analysis_text += """
\nTahmin Modeli Detayları:
1. Linear Regression
   - İlk 7 gün için %70 ağırlık
   - 7+ gün için %30 ağırlık
   - Kullanılan özellikler: Fiyat, Hacim ve Temel Teknik İndikatörler

2. XGBoost
   - İlk 7 gün için %30 ağırlık
   - 7+ gün için %70 ağırlık
   - Kullanılan özellikler: Tüm teknik indikatörler ve gelişmiş özellikler

Not: Bu tahminler geçmiş verilere dayalı matematiksel bir projeksiyon sunar.
Kripto para piyasaları yüksek volatiliteye sahiptir ve beklenmedik 
olaylar fiyatları önemli ölçüde etkileyebilir."""

            # Analiz penceresini göster
            dialog = AnalysisDialog(analysis_text, self)
            dialog.exec_()
            
            return future_df

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Gelecek tahmini hatası: {str(e)}")
            return pd.DataFrame()

    def update_technical_features(self, current_data, next_price, predictions):
        """Bir sonraki gün için teknik özellikleri güncelle"""
        current_data['Close'] = next_price
        current_data['Open'] = next_price
        current_data['High'] = next_price * 1.02
        current_data['Low'] = next_price * 0.98
        
        # Son n günlük tahminleri al
        recent_predictions = (predictions[-30:] if len(predictions) > 0 
                            else [next_price])
        
        # Teknik indikatörleri güncelle
        current_data['Returns'] = (next_price - predictions[-1]) / predictions[-1] if predictions else 0
        current_data['Volatility'] = np.std(recent_predictions) if len(recent_predictions) > 1 else 0
        current_data['MA7'] = np.mean(recent_predictions[-7:]) if len(recent_predictions) >= 7 else next_price
        current_data['MA30'] = np.mean(recent_predictions[-30:]) if len(recent_predictions) >= 30 else next_price
        current_data['MA90'] = next_price  # Yeterli veri olmadığı için basitleştirildi
        current_data['MA200'] = next_price  # Yeterli veri olmadığı için basitleştirildi
        
        # RSI ve diğer indikatörler için basitleştirilmiş güncellemeler
        current_data['RSI'] = 50  # Nötr değer
        current_data['MACD'] = 0
        current_data['Signal_Line'] = 0
        current_data['BB_middle'] = next_price
        current_data['BB_upper'] = next_price * 1.02
        current_data['BB_lower'] = next_price * 0.98

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())