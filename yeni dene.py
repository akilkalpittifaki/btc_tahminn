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
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dropout, Dense


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

    def calculate_new_features(self, predictions, current_price):
        """Yeni tahmin için özellikleri hesapla"""
        recent_prices = predictions[-7:] if len(predictions) >= 7 else predictions + [current_price]
        
        features = {
            'Open': current_price,
            'High': current_price * (1 + np.random.uniform(0, 0.02)),
            'Low': current_price * (1 - np.random.uniform(0, 0.02)),
            'Close': current_price,
            'Volume': self.df['Volume'].mean() * (1 + np.random.uniform(-0.2, 0.2)),
            'Returns': (current_price - predictions[-1])/predictions[-1] if predictions else 0,
            'Volatility': np.std(recent_prices) if len(recent_prices) > 1 else 0,
            'MA7': np.mean(recent_prices),
            'MA30': current_price,  # Basitleştirilmiş
            'MA90': current_price,  # Basitleştirilmiş
            'MA200': current_price,  # Basitleştirilmiş
            'RSI': 50 + np.random.uniform(-10, 10),  # Rastgele RSI
            'MACD': 0,  # Basitleştirilmiş
            'Signal_Line': 0,  # Basitleştirilmiş
            'BB_middle': current_price,
            'BB_upper': current_price * 1.02,
            'BB_lower': current_price * 0.98
        }
        
        return list(features.values())

    def calculate_rsi(self, prices, period=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        # LSTM için özellikler - daha fazla özellik ekledik
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                        'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                        'RSI', 'MACD', 'Signal_Line', 
                        'BB_middle', 'BB_upper', 'BB_lower']
        
        # Veriyi ölçeklendir
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Tüm özellikleri ölçeklendir
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
        self.y_scaled = self.price_scaler.fit_transform(self.df[['Close']])
        
        # LSTM için veri hazırlığı
        self.prepare_lstm_sequences()

    def prepare_lstm_training_data(self, time_step=60):
        """LSTM için zaman pencereli eğitim verilerini hazırla"""
        self.X_lstm = []
        self.y_lstm = []

        for i in range(time_step, len(self.y_scaled)):
            self.X_lstm.append(self.y_scaled[i-time_step:i, 0])
            self.y_lstm.append(self.y_scaled[i, 0])

        # LSTM için veriyi numpy array olarak kaydedin
        self.X_lstm, self.y_lstm = np.array(self.X_lstm), np.array(self.y_lstm)

        # LSTM giriş şekli (ornek, zaman adımı, özellik sayısı) - burada özellik sayısı 1
        self.X_lstm = np.reshape(self.X_lstm, (self.X_lstm.shape[0], self.X_lstm.shape[1], 1))

    def prepare_lstm_sequences(self, time_step=60):
        """LSTM için çok değişkenli zaman serisi verisi hazırla"""
        self.X_lstm = []
        self.y_lstm = []
        
        # Tüm özellikler için sekanslar oluştur
        for i in range(time_step, len(self.X_scaled)):
            self.X_lstm.append(self.X_scaled[i-time_step:i])
            # Sadece kapanış fiyatını hedef olarak al
            self.y_lstm.append(self.df['Close'].iloc[i])
        
        self.X_lstm = np.array(self.X_lstm)
        # y_lstm'i doğru şekilde ölçeklendir
        self.y_lstm = self.price_scaler.fit_transform(np.array(self.y_lstm).reshape(-1, 1))
        
        print(f"X_lstm shape: {self.X_lstm.shape}")
        print(f"y_lstm shape: {self.y_lstm.shape}")  
    
    def build_lstm_model(self):
        """Geliştirilmiş LSTM modeli"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.X_lstm.shape[1], self.X_lstm.shape[2])),
            Dropout(0.3),  # Dropout oranını %30 olarak artırdık
            LSTM(128, return_sequences=True),
            Dropout(0.3),  # Dropout oranını %30 olarak artırdık
            LSTM(64),
            Dropout(0.3),  # Dropout oranını %30 olarak artırdık
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model


    def train_models(self):
        """Modeli eğit"""
        self.lstm_model = self.build_lstm_model()
        
        # Early stopping ve model checkpoint ekle
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',  # Doğrulama kaybını izleyin
            patience=5,  # Eğer 5 epoch boyunca iyileşme olmazsa eğitim durdurulsun
            restore_best_weights=True  # En iyi ağırlıklarla geri yükleme
        )
        
        # Öğrenme oranı düşürme
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',  # Doğrulama kaybını izleyin
            factor=0.2,  # Öğrenme oranını %20'ye düşür
            patience=3,  # 3 epoch boyunca iyileşme olmazsa
            min_lr=0.0001  # Minimum öğrenme oranı
        )
        
        # Modeli eğit
        history = self.lstm_model.fit(
            self.X_lstm, self.y_lstm,
            epochs=100,  # Eğitim sayısını artırdık, ancak early stopping kullanıyoruz
            batch_size=16,
            validation_split=0.2,  # Verinin %20'sini doğrulama için kullan
            callbacks=[early_stopping, reduce_lr],  # Early stopping ve learning rate azaltma callback'leri
            verbose=1
        )

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
        """Geliştirilmiş gelecek tarih tahmini"""
        try:
            predictions = []
            dates = []
            
            # Son 60 günlük veriyi al
            last_sequence = self.X_scaled[-60:]
            print("Başlangıç sequence shape:", last_sequence.shape)
            
            for i in range(days_to_predict):
                # LSTM için giriş verisini hazırla
                lstm_input = last_sequence.reshape(1, 60, self.X_scaled.shape[1])
                print(f"Input shape: {lstm_input.shape}")
                
                # Tahmin yap
                pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)
                print(f"Raw prediction shape: {pred_scaled.shape}")
                
                # Tahmin değerini kapanış fiyatına dönüştür
                pred_scaled_reshaped = np.array([[float(pred_scaled[0][0])]])
                pred_price = self.price_scaler.inverse_transform(pred_scaled_reshaped)[0][0]
                print(f"Predicted price: {pred_price}")
                
                # Rastgelelik ekle
                volatility = np.std(self.df['Returns'].iloc[-30:]) * 300  # Artırıldı
                random_change = np.random.normal(0, volatility * 0.1)  # Rastgeleliği daha az etkili yapın
                pred_price *= (1 + random_change / 100)
                
                next_date = last_real_date + timedelta(days=i+1)
                predictions.append(pred_price)
                dates.append(next_date)
                
                # Yeni özellikleri hesapla ve numpy dizisine dönüştür
                features_dict = {
                    'Open': pred_price,
                    'High': pred_price * (1 + np.random.uniform(0, 0.02)),
                    'Low': pred_price * (1 - np.random.uniform(0, 0.02)),
                    'Close': pred_price,
                    'Volume': float(self.df['Volume'].mean() * (1 + np.random.uniform(-0.2, 0.2))),
                    'Returns': float((pred_price - predictions[-1])/predictions[-1] if len(predictions) > 0 else 0),
                    'Volatility': float(np.std(predictions[-7:]) if len(predictions) > 1 else 0),
                    'MA7': float(np.mean(predictions[-7:]) if len(predictions) >= 7 else pred_price),
                    'MA30': float(np.mean(predictions[-30:]) if len(predictions) >= 30 else pred_price),
                    'MA90': float(pred_price),
                    'MA200': float(pred_price),
                    'RSI': float(50 + np.random.uniform(-10, 10)),
                    'MACD': float(0),
                    'Signal_Line': float(0),
                    'BB_middle': float(pred_price),
                    'BB_upper': float(pred_price * 1.02),
                    'BB_lower': float(pred_price * 0.98)
                }
                
                # Sözlüğü sıralı bir listeye dönüştür
                new_features = [features_dict[feature] for feature in self.features]
                new_features = np.array(new_features, dtype=np.float32).reshape(1, -1)
                
                # Son sequence'i güncelle
                new_row_scaled = self.scaler.transform(new_features)
                last_sequence = np.vstack((last_sequence[1:], new_row_scaled[0]))
                
            df = pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions
            })
            print("Tahminler tamamlandı:", len(predictions), "gün")
            return df
            
        except Exception as e:
            print(f"Hata detayı: {str(e)}")
            print(f"Stack trace:")
            import traceback
            traceback.print_exc()
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

    def analyze_past_date(self, target_date):
        """Geçmiş tarih analizi"""
        try:
            # Hedef tarihe en yakın tarihi bul
            closest_date = self.df.index[self.df.index.get_indexer([target_date], method='nearest')[0]]
            
            # Gerçek değeri al
            actual_price = self.df.loc[closest_date, 'Close']
            
            # DataFrame oluştur
            return pd.DataFrame({
                'Tarih': [closest_date],
                'Tahmin': [actual_price]
            })
        except Exception as e:
            print(f"Geçmiş tarih analizi hatası: {str(e)}")
            return pd.DataFrame()        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
