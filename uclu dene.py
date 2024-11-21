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
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

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
            self.df['MA14'] = self.df['Close'].rolling(window=14).mean()
            self.df['MA30'] = self.df['Close'].rolling(window=30).mean()
            self.df['MA50'] = self.df['Close'].rolling(window=50).mean()
            self.df['MA90'] = self.df['Close'].rolling(window=90).mean()
            self.df['MA200'] = self.df['Close'].rolling(window=200).mean()
            
            # Momentum
            self.df['Momentum'] = self.df['Close'] - self.df['Close'].shift(10)
            
            # Stokastik Osilatör
            low_14 = self.df['Low'].rolling(window=14).min()
            high_14 = self.df['High'].rolling(window=14).max()
            self.df['Stochastic'] = 100 * (self.df['Close'] - low_14) / (high_14 - low_14)
            
            # Williams %R
            self.df['Williams_%R'] = 100 * (high_14 - self.df['Close']) / (high_14 - low_14)
            
            # Aroon Göstergesi
            aroon_period = 25
            self.df['Aroon_Up'] = 100 * (aroon_period - (aroon_period - self.df['High'].rolling(window=aroon_period).apply(lambda x: x.argmax() + 1))) / aroon_period
            self.df['Aroon_Down'] = 100 * (aroon_period - (aroon_period - self.df['Low'].rolling(window=aroon_period).apply(lambda x: x.argmin() + 1))) / aroon_period
            
            # Eksik değerleri temizle
            self.df.dropna(inplace=True)
            
        except Exception as e:
            print(f"Teknik indikatör hesaplama hatası: {str(e)}")
            raise e

    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        # LSTM için özellikler
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                        'Volatility', 'MA7', 'MA14', 'MA30', 'MA50', 'MA90', 'MA200', 
                        'Momentum', 'Stochastic', 'Williams_%R', 'Aroon_Up', 'Aroon_Down']
        
        # Özelliklerin veride mevcut olup olmadığını kontrol edin ve eksik olanları kaldırın
        self.features = [feature for feature in self.features if feature in self.df.columns]
        
        # Veriyi ölçeklendir
        self.scaler = StandardScaler()  # MinMaxScaler yerine StandardScaler kullandık
        self.price_scaler = StandardScaler()
        
        # Tüm özellikleri ölçeklendir
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
        self.y_scaled = self.price_scaler.fit_transform(self.df[['Close']])
        
        # LSTM için veri hazırlığı
        self.prepare_lstm_sequences()

        # XGBoost ve Random Forest için veri hazırlığı
        self.X_train = self.X_scaled
        self.y_train = self.df['Close'].values

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
        
    def build_lstm_model(self, input_shape, units, dropout_rate):
        """Geliştirilmiş LSTM modeli"""
        model = Sequential([
            LSTM(units, return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(units, return_sequences=True),
            Dropout(dropout_rate),
            LSTM(units // 2),
            Dropout(dropout_rate / 2),
            Dense(units // 4, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_models(self):
        """LSTM, XGBoost ve Random Forest modellerini eğit"""
        self.long_term_model = self.build_lstm_model(input_shape=(self.X_lstm.shape[1], self.X_lstm.shape[2]), units=128, dropout_rate=0.3)
        self.short_term_model = self.build_lstm_model(input_shape=(self.X_lstm.shape[1], self.X_lstm.shape[2]), units=64, dropout_rate=0.2)
        
        # Early stopping ve model checkpoint ekle
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=0.0001
        )
        
        # LSTM Modellerini eğit
        self.long_term_model.fit(
            self.X_lstm, self.y_lstm,
            epochs=100,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        self.short_term_model.fit(
            self.X_lstm, self.y_lstm,
            epochs=50,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # XGBoost Modelini eğit
        self.xgboost_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5)
        self.xgboost_model.fit(self.X_train, self.y_train)

        # Random Forest Modelini eğit
        self.random_forest_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.random_forest_model.fit(self.X_train, self.y_train)

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
            
            for i in range(days_to_predict):
                # LSTM için giriş verisini hazırla
                lstm_input = last_sequence.reshape(1, 60, last_sequence.shape[1])
                
                # Uzun ve kısa vadeli tahminler yap
                long_term_pred_scaled = self.long_term_model.predict(lstm_input, verbose=0)
                short_term_pred_scaled = self.short_term_model.predict(lstm_input, verbose=0)
                
                # XGBoost ve Random Forest tahminleri yap
                xgboost_pred = self.xgboost_model.predict(last_sequence[-1].reshape(1, -1))[0]
                random_forest_pred = self.random_forest_model.predict(last_sequence[-1].reshape(1, -1))[0]
                
                # Tahminleri birleştir, ağırlıklar ver
                pred_price = (0.3 * long_term_pred_scaled[0][0] +
                              0.3 * short_term_pred_scaled[0][0] +
                              0.2 * xgboost_pred +
                              0.2 * random_forest_pred)
                
                # Rastgelelik ekle
                volatility = np.std(self.df['Returns'].iloc[-30:]) * 0.1
                random_change = np.random.normal(0, volatility)
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
                
                # Sözlüğü sıralı bir listeye dönüştür ve numpy array'e çevir
                new_features = [features_dict[feature] for feature in self.features if feature in features_dict]
                new_features = np.array(new_features, dtype=np.float32).reshape(1, -1)
                
                # Özelliklerin sayısını scaler'ın beklediği özellik sayısına uyacak şekilde yeniden oluşturun
                if new_features.shape[1] < self.scaler.n_features_in_:
                    padding = np.zeros((1, self.scaler.n_features_in_ - new_features.shape[1]))
                    new_features = np.hstack((new_features, padding))
                
                # Özellikleri ölçeklendir
                new_row_scaled = self.scaler.transform(new_features)
                
                # Son sequence'i güncelle
                last_sequence = np.vstack((last_sequence[1:], new_row_scaled[0]))
            
            df = pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions
            })
            return df
            
        except Exception as e:
            print(f"Hata detayı: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Hata", f"Gelecek tahmini hatası: {str(e)}")
            return pd.DataFrame()
    
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
