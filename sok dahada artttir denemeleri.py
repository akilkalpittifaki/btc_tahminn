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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


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
        start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            self.add_technical_indicators()
            self.prepare_training_data()
            self.train_models()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def add_technical_indicators(self):
        """Teknik indikatörleri hesapla"""
        try:
            self.df['Returns'] = self.df['Close'].pct_change()
            self.df['Volatility'] = self.df['Returns'].rolling(window=20).std()
            self.df['MA7'] = self.df['Close'].rolling(window=7).mean()
            self.df['MA30'] = self.df['Close'].rolling(window=30).mean()
            self.df['MA90'] = self.df['Close'].rolling(window=90).mean()
            self.df['MA200'] = self.df['Close'].rolling(window=200).mean()
            self.df['RSI'] = self.calculate_rsi(self.df['Close'])
            exp1 = self.df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df['Close'].ewm(span=26, adjust=False).mean()
            self.df['MACD'] = exp1 - exp2
            self.df['Signal_Line'] = self.df['MACD'].ewm(span=9, adjust=False).mean()
            bb_window = 20
            rolling_mean = self.df['Close'].rolling(window=bb_window).mean()
            rolling_std = self.df['Close'].rolling(window=bb_window).std()
            self.df['BB_middle'] = rolling_mean
            self.df['BB_upper'] = rolling_mean + (rolling_std * 2)
            self.df['BB_lower'] = rolling_mean - (rolling_std * 2)
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
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                         'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                         'RSI', 'MACD', 'Signal_Line', 
                         'BB_middle', 'BB_upper', 'BB_lower']
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))
        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
        self.y_scaled = self.price_scaler.fit_transform(self.df[['Close']])
        self.prepare_lstm_sequences()

    def prepare_lstm_sequences(self, time_step=60):
        """LSTM için çok değişkenli zaman serisi verisi hazırla"""
        self.X_lstm = []
        self.y_lstm = []
        for i in range(time_step, len(self.X_scaled)):
            self.X_lstm.append(self.X_scaled[i-time_step:i])
            self.y_lstm.append(self.df['Close'].iloc[i])
        self.X_lstm = np.array(self.X_lstm)
        self.y_lstm = self.price_scaler.fit_transform(np.array(self.y_lstm).reshape(-1, 1))

    def build_lstm_model(self):
        """Geliştirilmiş LSTM modeli"""
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(self.X_lstm.shape[1], self.X_lstm.shape[2])),
            Dropout(0.2),
            LSTM(100, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='huber', metrics=['mae'])
        return model

    def train_models(self):
        """Modeli eğit"""
        self.lstm_model = self.build_lstm_model()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        history = self.lstm_model.fit(
            self.X_lstm, self.y_lstm,
            epochs=50,  # Eğitim epoch sayısı artırıldı
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

    def make_prediction(self):
        """Tahmin yap"""
        try:
            end_date = self.date_input.text()
            datetime.strptime(end_date, '%Y-%m-%d')
            predictions_df = self.predict_next_days(end_date)
            if predictions_df.empty:
                return
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2)
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--', linewidth=2)
            self.ax.legend()
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.grid(True)
            self.canvas.draw()
            self.results_text.clear()
            self.results_text.append("Tahmin Sonuçları:\n")
            for _, row in predictions_df.iterrows():
                self.results_text.append(f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\nTahmin: ${row['Tahmin']:.2f}\n\n")
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
            last_sequence = self.X_scaled[-60:]
            last_price = float(self.df['Close'].iloc[-1])
            price_30d_ago = float(self.df['Close'].iloc[-30])
            current_trend = (last_price - price_30d_ago) / price_30d_ago
            trend_momentum = current_trend * 2
            for i in range(days_to_predict):
                lstm_input = last_sequence.reshape(1, 60, self.X_scaled.shape[1])
                pred_scaled = self.lstm_model.predict(lstm_input, verbose=0)
                pred_scaled_reshaped = np.array([[float(pred_scaled[0][0])]])
                base_pred_price = float(self.price_scaler.inverse_transform(pred_scaled_reshaped)[0][0])
                short_term_volatility = np.std(self.df['Returns'].iloc[-30:]) * 400
                trend_effect = 1 + (trend_momentum * (i / days_to_predict))
                cycle_effect = np.sin(i / 10) * 0.1  # Artırıldı
                shock = 0
                if np.random.random() < 0.15:  # Şok olasılığı %10'dan %15'e çıkarıldı
                    shock = np.random.normal(0, short_term_volatility * 3)  # Şok büyüklüğü artırıldı
                daily_volatility = np.random.normal(0, short_term_volatility)
                total_effect = (1 + daily_volatility / 100) * trend_effect * (1 + cycle_effect) * (1 + shock / 100)
                pred_price = base_pred_price * total_effect
                next_date = last_real_date + timedelta(days=i + 1)
                predictions.append(pred_price)
                dates.append(next_date)
                new_features = [pred_price, pred_price * (1 + np.random.uniform(0, 0.05)),
                                pred_price * (1 - np.random.uniform(0, 0.05)), pred_price,
                                float(self.df['Volume'].mean() * np.random.uniform(0.5, 2.5)),
                                float((pred_price - predictions[-1]) / predictions[-1] if len(predictions) > 1 else 0),
                                float(np.std(predictions[-7:]) if len(predictions) > 1 else short_term_volatility),
                                float(np.mean(predictions[-7:]) if len(predictions) >= 7 else pred_price),
                                float(np.mean(predictions[-30:]) if len(predictions) >= 30 else pred_price),
                                float(pred_price), float(pred_price),
                                float(50 + np.random.uniform(-30, 30)),
                                float(trend_momentum * np.random.uniform(-2, 2)),
                                float(trend_momentum * np.random.uniform(-1, 1)),
                                float(pred_price),
                                float(pred_price * (1 + 0.05 * (1 + abs(trend_momentum)))),
                                float(pred_price * (1 - 0.05 * (1 + abs(trend_momentum))))]
                new_features = np.array(new_features, dtype=np.float32).reshape(1, -1)
                new_row_scaled = self.scaler.transform(new_features)
                last_sequence = np.vstack((last_sequence[1:], new_row_scaled[0]))
                if len(predictions) > 1:
                    short_trend = (predictions[-1] - predictions[-2]) / predictions[-2]
                    trend_momentum = trend_momentum * 0.95 + short_trend * 0.05
            df = pd.DataFrame({'Tarih': dates, 'Tahmin': predictions})
            return df
        except Exception as e:
            print(f"Hata detayı: {str(e)}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Hata", f"Gelecek tahmini hatası: {str(e)}")
            return pd.DataFrame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
