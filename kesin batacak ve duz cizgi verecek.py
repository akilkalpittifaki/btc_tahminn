import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QMessageBox,
                           QTextEdit, QSplitter)
from PyQt5.QtCore import Qt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import gc


class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # İlk olarak tüm değişkenleri tanımla
        self.lstm_model = None
        self.scaler = None
        self.price_scaler = None
        self.df = None
        self.X_scaled = None
        self.y_scaled = None
        self.X_lstm = None
        self.y_lstm = None

        # Temel özellikleri başlangıçta tanımla
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                        'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                        'RSI', 'MACD', 'Signal_Line', 
                        'BB_middle', 'BB_upper', 'BB_lower']

        self.ensemble_model = self.build_ensemble_model()  # Ensemble modeli başlat
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
        self.date_input.setPlaceholderText("Örnek: 2025-12-12")
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

    def build_ensemble_model(self):
        ensemble_model = {
            'rf_model': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'xgb_model': XGBRegressor(random_state=42, n_jobs=-1),
            'lgbm_model': LGBMRegressor(random_state=42, n_jobs=-1)
        }
        return ensemble_model

    def prepare_data_and_models(self):
        try:
            # Garbage collection
            gc.collect()

            self.ticker = "BTC-USD"
            start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')

            # Verileri indir
            self.df = yf.download(self.ticker, start=start_date, end=end_date)

            if self.df.empty:
                raise Exception("Veri indirilemedi")

            # Teknik indikatörler ekle
            self.add_technical_indicators()

            # Veriyi eğitim için hazırla
            self.prepare_training_data()

            # LSTM modelini eğit
            self.lstm_model = self.build_lstm_model()
            self.train_lstm_model()

            # Ensemble modelleri eğit
            self.train_ensemble_models()

        except Exception as e:
            error_msg = f"Veri hazırlama hatası: {str(e)}"
            QMessageBox.critical(self, "Hata", error_msg)
            raise

    def add_technical_indicators(self):
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
            raise ValueError(f"Teknik indikatör hesaplama hatası: {str(e)}")

    def calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_training_data(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.price_scaler = MinMaxScaler(feature_range=(0, 1))

        self.X_scaled = self.scaler.fit_transform(self.df[self.features])
        self.y_scaled = self.price_scaler.fit_transform(self.df[['Close']])

        self.prepare_lstm_sequences()

    def prepare_lstm_sequences(self, time_step=60):
        self.X_lstm = []
        self.y_lstm = []

        for i in range(time_step, len(self.X_scaled)):
            self.X_lstm.append(self.X_scaled[i-time_step:i])
            self.y_lstm.append(self.df['Close'].iloc[i])

        self.X_lstm = np.array(self.X_lstm)
        self.y_lstm = np.array(self.y_lstm).reshape(-1, 1)
        self.y_lstm = self.price_scaler.fit_transform(self.y_lstm)

    def build_lstm_model(self):
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

    def train_lstm_model(self):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

        self.lstm_model.fit(
            self.X_lstm, 
            self.y_lstm,
            epochs=30,
            batch_size=16,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

    def train_ensemble_models(self):
        X_2d = self.X_scaled
        y_2d = self.df['Close'].values

        for model_name, model in self.ensemble_model.items():
            model.fit(X_2d, y_2d)

    def plot_bitcoin_data(self):
        try:
            if self.df is None or self.df.empty:
                QMessageBox.warning(self, "Uyarı", "Gösterilecek veri yok!")
                return

            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Bitcoin Fiyatı', linewidth=2)
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            self.ax.legend()

            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            self.canvas.draw()
        except Exception as e:
            error_msg = f"Grafik çizim hatası: {str(e)}"
            QMessageBox.warning(self, "Uyarı", error_msg)

    def make_prediction(self):
        try:
            if not hasattr(self, 'lstm_model'):
                QMessageBox.warning(self, "Uyarı", "Model henüz eğitilmedi, lütfen bekleyin!")
                return

            if not self.date_input.text():
                QMessageBox.warning(self, "Uyarı", "Lütfen bir tarih girin!")
                return

            end_date = self.date_input.text()
            target_date = datetime.strptime(end_date, '%Y-%m-%d')

            if target_date < datetime.now():
                QMessageBox.warning(self, "Uyarı", "Lütfen gelecek bir tarih girin!")
                return

            predictions_df = self.predict_future_date(target_date)

            if predictions_df.empty:
                QMessageBox.warning(self, "Uyarı", "Tahmin oluşturulamadı!")
                return

            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2, color='blue')
            predictions_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, label='Tahminler', style='--', linewidth=2, color='red')

            self.ax.legend(loc='upper left')
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')

            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

            all_values = np.concatenate([self.df['Close'].values, predictions_df['Tahmin'].values])
            y_min = min(all_values) * 0.9
            y_max = max(all_values) * 1.1
            self.ax.set_ylim(y_min, y_max)

            self.canvas.draw()

            self.results_text.clear()
            self.results_text.append("Tahmin Sonuçları:\n\n")
            for _, row in predictions_df.iterrows():
                self.results_text.append(
                    f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"
                    f"Tahmin: ${row['Tahmin']:,.2f}\n\n"
                )

        except ValueError as ve:
            QMessageBox.warning(self, "Uyarı", f"Geçersiz tarih formatı: {str(ve)}")
        except Exception as e:
            error_msg = f"Tahmin hatası: {str(e)}"
            QMessageBox.critical(self, "Hata", error_msg)

    def predict_future_date(self, target_date):
        try:
            last_real_date = self.df.index[-1].tz_localize(None)
            days_to_predict = (target_date - last_real_date).days

            predictions = []
            dates = []

            last_sequence = self.X_scaled[-60:].copy()

            for i in range(days_to_predict):
                lstm_input = last_sequence.reshape(1, 60, -1)
                base_pred = self.lstm_model.predict(lstm_input, verbose=0).ravel()[0]
                base_pred = self.price_scaler.inverse_transform([[base_pred]])[0][0]

                next_date = last_real_date + timedelta(days=i + 1)
                predictions.append(base_pred)
                dates.append(next_date)

                new_features = last_sequence[-1].copy()
                new_features[0] = base_pred  # Close fiyatı olarak tahmini kullan
                last_sequence = np.vstack((last_sequence[1:], new_features))

            return pd.DataFrame({'Tarih': dates, 'Tahmin': predictions})
        except Exception as e:
            error_msg = f"Tahmin işlemi hatası: {str(e)}"
            print(error_msg)
            return pd.DataFrame()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
