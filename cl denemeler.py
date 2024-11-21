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
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import gc
from PyQt5.QtCore import QThread, pyqtSignal
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_predictor.log'),
        logging.StreamHandler()
    ]
)


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




class TrainingThread(QThread):
    finished = pyqtSignal()
    progress = pyqtSignal(int)
    error = pyqtSignal(str)
    
    def __init__(self, app_instance):
        super().__init__()
        self.app = app_instance
    
    def run(self):
        try:
            logging.info("Model eğitimi başlatılıyor...")
            
            # LSTM modelini eğit
            self.app.lstm_model = self.app.build_lstm_model()
            
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
            
            # LSTM'i eğit
            history = self.app.lstm_model.fit(
                self.app.X_lstm, 
                self.app.y_lstm,
                epochs=30,
                batch_size=16,
                validation_split=0.2,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
            
            # Diğer modeller için veri hazırla
            X_2d = self.app.X_scaled
            y_2d = self.app.df['Close'].values
            
            # Ensemble modelleri eğit
            logging.info("Ensemble modelleri eğitiliyor...")
            self.app.ensemble_model.lstm_model = self.app.lstm_model
            self.app.ensemble_model.rf_model.fit(X_2d, y_2d)
            self.app.ensemble_model.xgb_model.fit(X_2d, y_2d)
            self.app.ensemble_model.lgbm_model.fit(X_2d, y_2d)
            
            logging.info("Model eğitimi tamamlandı")
            self.finished.emit()
            
        except Exception as e:
            error_msg = f"Model eğitim hatası: {str(e)}"
            logging.error(error_msg)
            self.error.emit(error_msg)


class EnsembleModel:
    def __init__(self):
        self.lstm_model = None
        self.rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model = XGBRegressor(
            random_state=42,
            n_jobs=-1
        )
        self.lgbm_model = LGBMRegressor(
            random_state=42,
            n_jobs=-1
        )
            
    def prepare_data(self, sequence_data):
        """
        LSTM ve diğer modeller için veriyi hazırla
        sequence_data: shape (60, n_features) olan input
        """
        try:
            n_features = sequence_data.shape[1]
            
            # LSTM için veriyi (1, 60, n_features) şekline getir
            lstm_data = sequence_data.reshape(1, sequence_data.shape[0], n_features)
            
            # Diğer modeller için son time step'i al
            other_data = sequence_data[-1:].reshape(1, -1)
            
            return lstm_data, other_data
            
        except Exception as e:
            print(f"Veri hazırlama hatası: {str(e)}")
            print(f"sequence_data shape: {sequence_data.shape}")
            raise e
        
    def predict(self, sequence_data):
        try:
            lstm_data, other_data = self.prepare_data(sequence_data)
            
            # Model tahminleri
            lstm_pred = self.lstm_model.predict(lstm_data).flatten()
            rf_pred = self.rf_model.predict(other_data)
            xgb_pred = self.xgb_model.predict(other_data)
            lgbm_pred = self.lgbm_model.predict(other_data)
            
            # Ağırlıklı ortalama
            final_pred = (lstm_pred * 0.4 + 
                        rf_pred * 0.2 + 
                        xgb_pred * 0.2 + 
                        lgbm_pred * 0.2)
            
            return final_pred
            
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            print(f"Veri şekilleri:")
            print(f"sequence_data shape: {sequence_data.shape}")
            print(f"lstm_data shape: {lstm_data.shape if 'lstm_data' in locals() else 'not created'}")
            print(f"other_data shape: {other_data.shape if 'other_data' in locals() else 'not created'}")
            raise e



class BitcoinPredictorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Temel özellikleri başlangıçta tanımla
        self.features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 
                        'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                        'RSI', 'MACD', 'Signal_Line', 
                        'BB_middle', 'BB_upper', 'BB_lower']
        
        self.ensemble_model = EnsembleModel()  # Ensemble modeli başlat
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
        self.training_thread = None     

    def preprocess_data(self, df):
        """Gelişmiş veri önişleme"""
        try:
            logging.info("Veri önişleme başlatılıyor...")
            
            if df is None or df.empty:
                raise ValueError("Geçersiz veri")
            
            # Tarih indexini kontrol et
            if not isinstance(df.index, pd.DatetimeIndex):
                logging.warning("DataFrame'in index'i datetime değil, dönüştürülüyor...")
                df.index = pd.to_datetime(df.index)
            
            # NaN kontrolü ve temizleme
            initial_nulls = df.isnull().sum().sum()
            if initial_nulls > 0:
                logging.warning(f"{initial_nulls} adet NaN değer bulundu, temizleniyor...")
                
                # Önce forward fill, sonra backward fill
                df = df.fillna(method='ffill')
                df = df.fillna(method='bfill')
                
                # Hala NaN varsa ortalama ile doldur
                if df.isnull().sum().sum() > 0:
                    df = df.fillna(df.mean())
                
                final_nulls = df.isnull().sum().sum()
                if final_nulls > 0:
                    raise ValueError(f"Temizleme sonrası hala {final_nulls} adet NaN mevcut")
            
            # Aykırı değer kontrolü
            for column in df.select_dtypes(include=[np.number]).columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
                if outliers > 0:
                    logging.warning(f"{column} sütununda {outliers} adet aykırı değer bulundu")
            
            logging.info("Veri önişleme tamamlandı")
            return df
            
        except Exception as e:
            error_msg = f"Veri önişleme hatası: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)


    def prepare_data_and_models(self):
        """Veri hazırlama ve model eğitimi"""
        try:
            self.ticker = "BTC-USD"
            # 4 yıllık veri al
            start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Verileri indir
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            
            if self.df.empty:
                raise Exception("Veri indirilemedi")
                
            # Teknik indikatörler ekle
            self.add_technical_indicators()
            
            # Trend bileşenlerini ekle
            self.add_trend_components()
            
            # Veriyi eğitim için hazırla
            self.prepare_training_data()
            
            # Modelleri eğit
            self.train_models()
            
        except Exception as e:
            error_msg = f"Veri hazırlama hatası: {str(e)}"
            QMessageBox.critical(self, "Hata", error_msg)
            raise

    def add_trend_components(self):
        """Uzun ve kısa vadeli trend bileşenleri ekle"""
        # Halving cycle (yaklaşık 4 yıl) etkisi
        self.halving_cycle = 4 * 365
        last_halving = datetime(2024, 4, 1)  # Son halving tarihi
        days_since_last_halving = (datetime.now() - last_halving).days
        self.halving_effect = np.sin(2 * np.pi * days_since_last_halving / self.halving_cycle)
        
        # Market döngüsü (yaklaşık 2 yıl)
        self.market_cycle = 2 * 365
        self.market_phase = np.sin(2 * np.pi * days_since_last_halving / self.market_cycle)
        
        # Momentum göstergeleri
        self.momentum_short = self.df['Close'].pct_change(30).mean()  # 30 günlük momentum
        self.momentum_long = self.df['Close'].pct_change(90).mean()   # 90 günlük momentum
        
        # DataFrame'e yeni özellikleri ekle
        self.df['Halving_Effect'] = self.halving_effect
        self.df['Market_Phase'] = self.market_phase
        self.df['Momentum_Short'] = self.df['Close'].pct_change(30)
        self.df['Momentum_Long'] = self.df['Close'].pct_change(90)
        
        # Trend güç göstergesi
        self.df['Trend_Strength'] = (self.df['Momentum_Short'] + self.df['Momentum_Long']) / 2
        
        # Eksik değerleri doldur
        self.df.fillna(method='bfill', inplace=True)
        
        # Özellikleri features listesine ekle
        self.features.extend(['Halving_Effect', 'Market_Phase', 'Momentum_Short', 
                            'Momentum_Long', 'Trend_Strength'])

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

    def _calculate_advanced_features(self, predictions, current_price, trend_momentum=0):
        """Gelişmiş teknik gösterge hesaplamaları"""
        try:
            # current_price'ı float'a çevir ve NaN kontrolü
            if isinstance(current_price, pd.Series):
                current_price = float(current_price.iloc[0])
            elif isinstance(current_price, np.ndarray):
                current_price = float(current_price[0])
            else:
                current_price = float(current_price)
            
            if np.isnan(current_price):
                current_price = float(self.df['Close'].iloc[-1])
            
            # Son fiyatları al ve numpy array'e çevir
            if len(predictions) >= 7:
                recent_prices = np.array([float(p) if isinstance(p, (pd.Series, np.ndarray)) else float(p) 
                                        for p in predictions[-7:]], dtype=np.float64)
            else:
                temp_predictions = []
                for p in predictions:
                    if isinstance(p, pd.Series):
                        temp_predictions.append(float(p.iloc[0]))
                    elif isinstance(p, np.ndarray):
                        temp_predictions.append(float(p[0]))
                    else:
                        temp_predictions.append(float(p))
                temp_predictions.append(current_price)
                recent_prices = np.array(temp_predictions, dtype=np.float64)
            
            # Volatilite hesapla
            if len(recent_prices) > 1:
                volatility = float(np.std(recent_prices))
            else:
                volatility = 0.02
            
            # Trend marjı hesapla
            trend_margin = 0.02 * (1 + abs(trend_momentum))
            
            # RSI hesaplamaları
            rsi_base = 50
            rsi_momentum = trend_momentum * 25
            rsi_noise = np.random.uniform(-10, 10)
            rsi_value = np.clip(rsi_base + rsi_momentum + rsi_noise, 0, 100)
            
            # MACD hesaplamaları
            macd_base = trend_momentum * volatility * current_price * 0.01
            macd_noise = np.random.normal(0, volatility * current_price * 0.005)
            
            # Bollinger bant genişliği
            bb_width = max(0.02, volatility * 2)
            
            # MA hesaplamaları için liste hazırla
            pred_list = []
            for p in predictions:
                if isinstance(p, pd.Series):
                    pred_list.append(float(p.iloc[0]))
                elif isinstance(p, np.ndarray):
                    pred_list.append(float(p[0]))
                else:
                    pred_list.append(float(p))
            
            # Özellik sözlüğünü hazırla
            features = {
                'Open': float(current_price * (1 + np.random.uniform(-0.005, 0.005))),
                'High': float(current_price * (1 + trend_margin + np.random.uniform(0, 0.01))),
                'Low': float(current_price * (1 - trend_margin - np.random.uniform(0, 0.01))),
                'Close': float(current_price),
                'Volume': float(self.df['Volume'].mean() * np.random.uniform(0.5, 2.0)),
                'Returns': float((current_price - pred_list[-1])/pred_list[-1] if pred_list and len(pred_list) > 0 else 0),
                'Volatility': float(volatility),
                'MA7': float(np.mean(recent_prices)),
                'MA30': float(np.mean(pred_list[-30:] + [current_price] if len(pred_list) >= 30 else [current_price])),
                'MA90': float(np.mean(pred_list[-90:] + [current_price] if len(pred_list) >= 90 else [current_price])),
                'MA200': float(np.mean(pred_list[-200:] + [current_price] if len(pred_list) >= 200 else [current_price])),
                'RSI': float(rsi_value),
                'MACD': float(macd_base + macd_noise),
                'Signal_Line': float(macd_base),
                'BB_middle': float(current_price),
                'BB_upper': float(current_price * (1 + bb_width)),
                'BB_lower': float(current_price * (1 - bb_width)),
                'Halving_Effect': float(self.halving_effect if isinstance(self.halving_effect, (int, float)) else self.halving_effect.iloc[0]),
                'Market_Phase': float(self.market_phase if isinstance(self.market_phase, (int, float)) else self.market_phase.iloc[0]),
                'Momentum_Short': float(self.momentum_short if isinstance(self.momentum_short, (int, float)) else self.momentum_short.iloc[0]),
                'Momentum_Long': float(self.momentum_long if isinstance(self.momentum_long, (int, float)) else self.momentum_long.iloc[0]),
                'Trend_Strength': float(trend_momentum)
            }
            
            # Features listesinde olmayan özellikleri kontrol et
            missing_features = set(self.features) - set(features.keys())
            if missing_features:
                for feature in missing_features:
                    features[feature] = float(current_price)
            
            # Tüm değerlerin float olduğundan emin ol
            return {feature: float(value) for feature, value in features.items()}
                
        except Exception as e:
            print(f"Feature calculation error: {str(e)}")
            print(f"Current price: {current_price}")
            print(f"Predictions: {predictions}")
            raise

    def calculate_rsi(self, prices, period=14):
        """RSI hesapla"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        try:
            # Eksik değerleri doldur
            for feature in self.features:
                if feature in self.df.columns:
                    self.df[feature].fillna(self.df[feature].bfill(), inplace=True)
            
            # Veriyi ölçeklendir
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.price_scaler = MinMaxScaler(feature_range=(0, 1))
            
            # Tüm özellikleri ölçeklendir
            self.X_scaled = self.scaler.fit_transform(self.df[self.features])
            self.y_scaled = self.price_scaler.fit_transform(self.df[['Close']])
            
            # LSTM için veri hazırlığı
            self.prepare_lstm_sequences()
            
        except Exception as e:
            error_msg = f"Veri hazırlama hatası: {str(e)}\n"
            error_msg += "Mevcut özellikler:\n"
            error_msg += str(self.df.columns.tolist())
            raise Exception(error_msg)
    
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
        try:
            self.X_lstm = []
            self.y_lstm = []
            
            # Eksik değerleri kontrol et ve doldur
            if self.X_scaled is None or len(self.X_scaled) == 0:
                raise ValueError("X_scaled verisi boş veya None")
                
            # Tüm özellikler için sekanslar oluştur
            for i in range(time_step, len(self.X_scaled)):
                self.X_lstm.append(self.X_scaled[i-time_step:i])
                self.y_lstm.append(self.df['Close'].iloc[i])
            
            self.X_lstm = np.array(self.X_lstm)
            self.y_lstm = np.array(self.y_lstm).reshape(-1, 1)
            self.y_lstm = self.price_scaler.fit_transform(self.y_lstm)
            
            print(f"X_lstm shape: {self.X_lstm.shape}")
            print(f"y_lstm shape: {self.y_lstm.shape}")
            
        except Exception as e:
            print(f"LSTM sequence hazırlama hatası: {str(e)}")
            raise  
    
    def build_lstm_model(self):
        """Geliştirilmiş LSTM modeli input shape kontrolü ile"""
        try:
            if not hasattr(self, 'X_lstm') or self.X_lstm is None:
                raise ValueError("LSTM verisi hazır değil")
                
            input_shape = self.X_lstm.shape
            if len(input_shape) != 3:
                raise ValueError(f"Geçersiz input shape: {input_shape}, (samples, time steps, features) bekleniyordu")
                
            logging.info(f"LSTM modeli oluşturuluyor, input shape: {input_shape}")
            
            model = Sequential([
                LSTM(100, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.2),
                LSTM(100, return_sequences=True),
                Dropout(0.2),
                LSTM(50),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='huber', metrics=['mae'])
            logging.info("LSTM modeli başarıyla oluşturuldu")
            
            return model
            
        except Exception as e:
            error_msg = f"LSTM model oluşturma hatası: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def train_models(self):
        """Thread kullanarak modelleri eğit"""
        try:
            if self.training_thread is not None and self.training_thread.isRunning():
                logging.warning("Model eğitimi zaten devam ediyor")
                return
                
            self.training_thread = TrainingThread(self)
            self.training_thread.finished.connect(self.on_training_finished)
            self.training_thread.error.connect(self.on_training_error)
            
            # UI'yi devre dışı bırak
            self.predict_button.setEnabled(False)
            
            logging.info("Model eğitimi başlatılıyor...")
            self.training_thread.start()
            
        except Exception as e:
            error_msg = f"Model eğitimi başlatma hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Hata", error_msg)

    def on_training_finished(self):
        """Eğitim tamamlandığında çağrılır"""
        self.predict_button.setEnabled(True)
        logging.info("Model eğitimi başarıyla tamamlandı")
        QMessageBox.information(self, "Bilgi", "Model eğitimi tamamlandı")

    def on_training_error(self, error_msg):
        """Eğitim hatası durumunda çağrılır"""
        self.predict_button.setEnabled(True)
        logging.error(f"Model eğitimi hatası: {error_msg}")
        QMessageBox.critical(self, "Hata", f"Model eğitimi hatası: {error_msg}")                    

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
        """
        Gelecek tarihleri tahmin eden gelişmiş fonksiyon.
        
        Args:
            days_to_predict (int): Tahmin edilecek gün sayısı
            last_real_date (datetime): Son gerçek veri tarihi
            
        Returns:
            pd.DataFrame: Tahmin sonuçlarını içeren DataFrame
        """
        try:
            if not hasattr(self, 'scaler') or self.scaler is None:
                raise ValueError("Scaler initialization error")
                
            if self.X_scaled is None or len(self.X_scaled) < 60:
                raise ValueError("Insufficient historical data")
                
            predictions = []
            dates = []
            trends = []
            confidences = []
            
            # Trend fazları ve özellikleri
            trend_phases = {
                'bull': {
                    'duration': (60, 90),
                    'strength': (0.3, 0.8),
                    'volatility': (0.02, 0.04)
                },
                'bear': {
                    'duration': (30, 60),
                    'strength': (-0.5, -0.2),
                    'volatility': (0.03, 0.05)
                },
                'accumulation': {
                    'duration': (20, 40),
                    'strength': (-0.1, 0.1),
                    'volatility': (0.01, 0.02)
                },
                'distribution': {
                    'duration': (15, 30),
                    'strength': (-0.3, 0.3),
                    'volatility': (0.02, 0.03)
                }
            }
            
            # Son sequence'i al ve doğrula
            last_sequence = self.X_scaled[-60:].copy()
            if len(last_sequence) != 60:
                raise ValueError(f"Invalid sequence length: {len(last_sequence)}")
                
            # Başlangıç değerlerini ayarla
            current_price = float(self.df['Close'].iloc[-1])
            current_trend = 'accumulation'  # Başlangıç trendi
            trend_days_left = np.random.randint(*trend_phases[current_trend]['duration'])
            
            # Market döngüsü parametreleri
            halving_cycle_days = 4 * 365  # ~4 yıllık halving cycle
            market_cycle_days = 365  # ~1 yıllık market cycle
            
            for i in range(days_to_predict):
                try:
                    # Trend değişimi kontrolü
                    if trend_days_left <= 0:
                        if current_trend == 'accumulation':
                            current_trend = 'bull' if np.random.random() > 0.3 else 'distribution'
                        elif current_trend == 'bull':
                            current_trend = 'distribution' if np.random.random() > 0.2 else 'bull'
                        elif current_trend == 'distribution':
                            current_trend = 'bear'
                        else:  # bear
                            current_trend = 'accumulation'
                        
                        trend_days_left = np.random.randint(*trend_phases[current_trend]['duration'])
                    
                    # Cycle etkileri
                    days_since_start = i + (datetime.now() - datetime(2024, 4, 1)).days  # Son halving'den beri geçen gün
                    halving_effect = np.sin(2 * np.pi * days_since_start / halving_cycle_days)
                    market_cycle_effect = np.sin(2 * np.pi * days_since_start / market_cycle_days)
                    
                    # Model tahmini
                    lstm_input = last_sequence.reshape(1, 60, -1)
                    try:
                        base_pred = self.ensemble_model.predict(last_sequence)
                        if isinstance(base_pred, (list, np.ndarray)):
                            base_pred = base_pred[0]
                    except Exception as model_error:
                        print(f"Model tahmin hatası: {str(model_error)}")
                        base_pred = current_price
                    
                    # Trend etkileri
                    trend_strength = np.random.uniform(*trend_phases[current_trend]['strength'])
                    trend_volatility = np.random.uniform(*trend_phases[current_trend]['volatility'])
                    
                    # Momentum ve cycle etkileri
                    momentum_effect = np.mean([(p - predictions[max(0, i-j)]) / predictions[max(0, i-j)] 
                                            if i >= j and predictions 
                                            else 0 
                                            for j in [1, 3, 7]]) if predictions else 0
                    
                    cycle_effect = halving_effect * 0.3 + market_cycle_effect * 0.2
                    
                    # Rastgele şoklar
                    shock = np.random.normal(0, trend_volatility * 2) if np.random.random() < 0.05 else 0
                    
                    # Final tahmin hesaplama
                    pred_price = base_pred * (1 + trend_strength)
                    pred_price *= (1 + cycle_effect * 0.1)
                    pred_price *= (1 + momentum_effect * 0.2)
                    pred_price *= (1 + shock)
                    
                    # Makul tahmin aralığı kontrolü
                    max_daily_change = 0.15  # Maksimum %15 günlük değişim
                    min_price = current_price * (1 - max_daily_change)
                    max_price = current_price * (1 + max_daily_change)
                    pred_price = np.clip(pred_price, min_price, max_price)
                    
                    # NaN kontrolü
                    if np.isnan(pred_price):
                        pred_price = current_price
                    
                    # Tahmin güvenilirliği hesaplama
                    confidence = 1.0 - (i / days_to_predict) * 0.5  # Zaman ilerledikçe güven azalır
                    confidence *= 1.0 - abs(trend_strength)  # Trend gücü arttıkça güven azalır
                    confidence *= 1.0 - trend_volatility  # Volatilite arttıkça güven azalır
                    
                    # Sonuçları kaydet
                    next_date = last_real_date + timedelta(days=i+1)
                    predictions.append(float(pred_price))
                    dates.append(next_date)
                    trends.append(current_trend)
                    confidences.append(float(confidence))
                    
                    # Yeni özellikleri hesapla ve güncelle
                    try:
                        features_dict = self._calculate_advanced_features(
                            predictions[-10:] if predictions else [current_price],
                            pred_price,
                            trend_momentum=trend_strength
                        )
                        new_features = np.array([float(features_dict[feature]) for feature in self.features])
                        new_row_scaled = self.scaler.transform(new_features.reshape(1, -1))
                        
                        # Son sequence'i güncelle
                        last_sequence = np.vstack((last_sequence[1:], new_row_scaled))
                    except Exception as feature_error:
                        print(f"Özellik güncelleme hatası: {str(feature_error)}")
                    
                    current_price = pred_price
                    trend_days_left -= 1
                    
                    # Bellek optimizasyonu
                    if i % 30 == 0:
                        gc.collect()
                    
                except Exception as iteration_error:
                    print(f"İterasyon hatası (gün {i}): {str(iteration_error)}")
                    continue
            
            # Sonuçları DataFrame'e dönüştür
            results_df = pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions,
                'Trend': trends,
                'Güven': confidences
            })
            
            # Son temizlik
            gc.collect()
            
            return results_df
            
        except Exception as e:
            error_msg = f"Tahmin işlemi hatası: {str(e)}\n"
            print(error_msg)
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

    def cleanup(self):
        """Bellek temizleme"""
        try:
            if hasattr(self, 'lstm_model'):
                del self.lstm_model
            if hasattr(self, 'ensemble_model'):
                del self.ensemble_model
            if hasattr(self, 'X_lstm'):
                del self.X_lstm
            if hasattr(self, 'y_lstm'):
                del self.y_lstm
            if hasattr(self, 'X_scaled'):
                del self.X_scaled
            
            # Tensorflow backend temizliği
            tf.keras.backend.clear_session()
            
            # Garbage collection
            gc.collect()
            
        except Exception as e:
            logging.error(f"Cleanup hatası: {str(e)}")
     

        

        


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
