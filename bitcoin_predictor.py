# Sistem ve veri işleme ile ilgili kütüphaneler
import sys
import gc
import logging
from datetime import datetime, timedelta

# Veri analiz ve bilimsel hesaplama kütüphaneleri
import numpy as np
import pandas as pd
import yfinance as yf

# Sklearn ve makine öğrenmesi ile ilgili kütüphaneler
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# TensorFlow ve Keras katmanları
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# PyQt5 - GUI ve widgetlar
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QPushButton, QLabel, QLineEdit, QMessageBox,
    QTextEdit, QSplitter, QDialog
)

# Matplotlib - Grafik çizim araçları
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import YearLocator, DateFormatter



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
        try:
            # Boyut düzeltmeleri
            sequence_data = np.array(sequence_data)
            if len(sequence_data.shape) == 1:
                sequence_data = sequence_data.reshape(1, -1)
                
            return sequence_data, sequence_data
            
        except Exception as e:
            print(f"Veri hazırlama hatası: {str(e)}")
            raise e
            
    def predict(self, sequence_data):
        try:
            if not isinstance(sequence_data, np.ndarray):
                sequence_data = np.array(sequence_data)
                
            # Sadece bir tahmin döndür
            return float(self.lstm_model.predict(sequence_data.reshape(1, -1, sequence_data.shape[-1]), verbose=0)[0])
            
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
            raise e


class BitcoinPredictorApp(QMainWindow):

    def changeEvent(self, event):
        """Pencere durumu değiştiğinde çağrılır"""
        if event.type() == QEvent.WindowStateChange:
            if self.windowState() != Qt.WindowMaximized:
                self.showMaximized()
        super().changeEvent(event)

    def resizeEvent(self, event):
        """Pencere boyutu değiştiğinde çağrılır"""
        super().resizeEvent(event)
        # Pencereyi tam ekranda tut
        if not self.isMaximized():
            self.showMaximized()


    def __init__(self):
        super().__init__()
        
        # Pencereyi tam ekran ve maksimize edilmiş durumda tut
        self.setWindowState(Qt.WindowMaximized)
        self.setWindowFlags(
            Qt.Window |
            Qt.CustomizeWindowHint |
            Qt.WindowTitleHint |
            Qt.WindowSystemMenuHint |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )
        
        # Pencereyi göster ve maximize et
        self.show()
        self.showMaximized()
        
        # Minimum pencere boyutunu ekran boyutuna eşitle
        screen = QApplication.primaryScreen().geometry()
        self.setMinimumSize(screen.width(), screen.height())
        
        # İlk olarak tüm değişkenleri tanımla
        self.training_thread = None
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
        
        self.ensemble_model = EnsembleModel()  # Ensemble modeli başlat
        
        # Pencerenin başlığını ayarla
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")
        
        # Ana widget ve layout oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Grafik boyutunu büyüt
        self.figure, self.ax = plt.subplots(figsize=(20, 12))  # Grafik boyutunu artırdım
        
        # Sol taraf için düzen
        splitter = QSplitter(Qt.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Canvas ekle
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)
        
        # Tarih girişi için widget
        date_widget = QWidget()
        date_layout = QVBoxLayout(date_widget)
        
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")
        self.date_label.setStyleSheet("""
            QLabel {
                font-size: 14pt;
                font-weight: bold;
                color: #2C3E50;
                margin-bottom: 5px;
            }
        """)
        
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")
        self.date_input.setFixedWidth(400)  # Genişliği artırdım
        self.date_input.setStyleSheet("""
            QLineEdit {
                font-size: 14pt;
                padding: 8px;
                border: 2px solid #BDC3C7;
                border-radius: 6px;
                background-color: white;
            }
            QLineEdit:focus {
                border: 2px solid #3498DB;
            }
        """)
        
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_input)
        
        # Tahmin butonu
        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.clicked.connect(self.make_prediction)
        self.predict_button.setFixedWidth(400)  # Genişliği artırdım
        self.predict_button.setFixedHeight(50)  # Yükseklik ekledim
        self.predict_button.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                background-color: #2ECC71;
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
            QPushButton:pressed {
                background-color: #229954;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        
        date_layout.addWidget(self.predict_button)
        date_layout.setSpacing(15)  # Widget'lar arası boşluk
        left_layout.addWidget(date_widget)
        
        # Sonuçlar için text widget
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(500)  # Genişliği artırdım
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-size: 13pt;
                padding: 10px;
                border: 2px solid #BDC3C7;
                border-radius: 6px;
                background-color: white;
                color: #2C3E50;
            }
        """)
        
        # Widget'ları splitter'a ekle
        splitter.addWidget(left_widget)
        splitter.addWidget(self.results_text)
        
        # Splitter oranlarını ayarla (75% grafik, 25% sonuçlar)
        main_layout.addWidget(splitter)
        splitter.setSizes([750, 250])
        
        # Grafik fontlarını ve stilini ayarla
        plt.style.use('classic')  # Klasik stil kullan
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2.5,
            'figure.autolayout': True,
            'axes.facecolor': '#f0f0f0',
            'grid.color': 'white',
            'grid.linestyle': '-',
            'grid.linewidth': 1,
            'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        })
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 16,
            'axes.labelsize': 14,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2.5
        })
        
        # Başlangıç verilerini hazırla ve modeli eğit
        self.prepare_data_and_models()
        self.plot_bitcoin_data()

        self.evaluate_button = QPushButton("Modeli Değerlendir")
        self.evaluate_button.setFixedWidth(400)  # Genişliği tahmin butonuyla aynı
        self.evaluate_button.setFixedHeight(50)  # Yükseklik tahmin butonuyla aynı
        self.evaluate_button.clicked.connect(self.evaluate_model_performance)
        self.evaluate_button.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                background-color: #3498DB;  # Mavi renk
                color: white;
                border: none;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
            QPushButton:pressed {
                background-color: #2475A8;
            }
            QPushButton:disabled {
                background-color: #BDC3C7;
            }
        """)
        
        date_layout.addWidget(self.evaluate_button)   


    def evaluate_model_performance(self):
        """Model performans değerlendirmesi"""
        try:
            # Model eğitilmemiş ise uyar
            if not hasattr(self, 'lstm_model') or self.lstm_model is None:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce modeli eğitin!")
                return

            # Test verisi oluştur (son 30 gün)
            test_start_date = self.df.index[-30]
            test_data = self.df[self.df.index >= test_start_date].copy()
            
            # UTC bilgisini kaldır
            if hasattr(test_data.index, 'tz'):
                test_data.index = test_data.index.tz_localize(None)
            
            # Test tarihleri için tahminler al
            predictions = self.predict_next_days(test_data.index[-1].strftime('%Y-%m-%d'))
            
            if predictions.empty:
                QMessageBox.warning(self, "Uyarı", "Tahmin oluşturulamadı!")
                return
            
            # Değerlendirici oluştur
            evaluator = ModelEvaluator(self)
            
            # Metrikleri hesapla
            evaluator.calculate_metrics(test_data, predictions)
            evaluator.calculate_profit_metrics(test_data, predictions)
            
            # Rapor oluştur ve göster
            report = evaluator.generate_evaluation_report()
            
            # Sonuçları göster
            dialog = AnalysisDialog(report, self)
            dialog.exec_()

        except Exception as e:
            error_msg = f"Model değerlendirme hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Hata", error_msg)       
        
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
            # Garbage collection
            gc.collect()      

            self.ticker = "BTC-USD"
            # 4 yıllık veri al
            start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Verileri indir
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            
            if self.df.empty:
                raise Exception("Veri indirilemedi")
            
            # DataFrame'in bir kopyasını al
            self.df = self.df.copy()            
                
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
            # Tahminleri float'a çevir
            predictions = [float(p) if isinstance(p, (pd.Series, np.ndarray)) else float(p) 
                        for p in predictions]
            
            # Current price'ı float'a çevir
            if isinstance(current_price, pd.Series):
                current_price = float(current_price.iloc[0])
            elif isinstance(current_price, np.ndarray):
                current_price = float(current_price[0])
            else:
                current_price = float(current_price)
            
            trend_momentum = float(trend_momentum)
            
            # NaN kontrolü
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
            
            # Volume hesaplama
            mean_volume = float(self.df['Volume'].mean())
            
            # Özellik sözlüğünü hazırla
            features = {
                'Open': float(current_price * (1 + np.random.uniform(-0.005, 0.005))),
                'High': float(current_price * (1 + trend_margin + np.random.uniform(0, 0.01))),
                'Low': float(current_price * (1 - trend_margin - np.random.uniform(0, 0.01))),
                'Close': float(current_price),
                'Volume': float(mean_volume * np.random.uniform(0.5, 2.0)),
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
                'Halving_Effect': float(self.halving_effect if isinstance(self.halving_effect, (int, float)) else self.halving_effect),
                'Market_Phase': float(self.market_phase if isinstance(self.market_phase, (int, float)) else self.market_phase),
                'Momentum_Short': float(self.momentum_short if isinstance(self.momentum_short, (int, float)) else self.momentum_short),
                'Momentum_Long': float(self.momentum_long if isinstance(self.momentum_long, (int, float)) else self.momentum_long),
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
        try:
            # DataFrame kontrolü
            if self.df is None or self.df.empty:
                raise ValueError("Veri seti boş veya yüklenemedi")
                
            # Minimum veri kontrolü
            if len(self.df) < 60:
                raise ValueError("Yeterli veri yok (minimum 60 gün gerekli)")
            
            # DataFrame'in bir kopyasını oluştur
            self.df = self.df.copy()            
            
            # Eksik değerleri doldur
            for feature in self.features:
                if feature in self.df.columns:
                    self.df[feature] = self.df[feature].fillna(method='bfill')
            
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
            logging.error(error_msg)
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
        """Geliştirilmiş LSTM modeli"""
        try:
            if not hasattr(self, 'X_lstm') or self.X_lstm is None:
                raise ValueError("LSTM verisi hazır değil")
                
            input_shape = self.X_lstm.shape
            if len(input_shape) != 3:
                raise ValueError(f"Geçersiz input shape: {input_shape}")
                
            model = Sequential([
                LSTM(150, return_sequences=True, input_shape=(input_shape[1], input_shape[2])),
                Dropout(0.3),
                LSTM(100, return_sequences=True),
                Dropout(0.3),
                LSTM(100, return_sequences=True),
                Dropout(0.3),
                LSTM(50),
                Dropout(0.2),
                Dense(50, activation='relu'),
                Dense(25, activation='relu'),
                Dense(1, activation='linear')
            ])
            
            # Düzeltilmiş loss fonksiyonu
            def custom_loss(y_true, y_pred):
                # tf.keras yerine keras kullanılıyor
                mse = tf.reduce_mean(tf.square(y_true - y_pred))
                decline_penalty = tf.reduce_mean(tf.maximum(0., y_true - y_pred) * 2)
                return mse + decline_penalty
            
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
            model.compile(optimizer=optimizer, 
                        loss=custom_loss,  # Custom loss kullanıyoruz
                        metrics=['mae'])
            
            return model
            
        except Exception as e:
            error_msg = f"LSTM model oluşturma hatası: {str(e)}"
            logging.error(error_msg)
            raise ValueError(error_msg)

    def train_models(self):
        try:
            # Önceki modeli temizle
            if hasattr(self, 'lstm_model'):
                del self.lstm_model
                tf.keras.backend.clear_session()
                gc.collect()

            if not hasattr(self, 'training_thread') or self.training_thread is None:
                self.training_thread = TrainingThread(self)
                self.training_thread.finished.connect(self.on_training_finished)
                self.training_thread.error.connect(self.on_training_error)
                
                # UI'yi devre dışı bırak
                self.predict_button.setEnabled(False)
                
                logging.info("Model eğitimi başlatılıyor...")
                self.training_thread.start()
            elif self.training_thread.isRunning():
                logging.warning("Model eğitimi zaten devam ediyor")
                QMessageBox.warning(self, "Uyarı", "Model eğitimi devam ediyor, lütfen bekleyin.")
            else:
                self.training_thread = TrainingThread(self)
                self.training_thread.finished.connect(self.on_training_finished)
                self.training_thread.error.connect(self.on_training_error)
                self.predict_button.setEnabled(False)
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
        try:
            if self.df is None or self.df.empty:
                QMessageBox.warning(self, "Uyarı", "Gösterilecek veri yok!")
                return
            
            self.ax.clear()
            
            # Tarihleri datetime dizisi olarak al
            dates = self.df.index.to_pydatetime()
            
            # Veri aralığını belirle
            prices = self.df['Close'].values
            min_price = np.min(prices)
            max_price = np.max(prices)
            price_range = max_price - min_price
            
            # Y eksen limitlerini ayarla - %20 margin ekle
            y_min = max(0, min_price - price_range * 0.2)
            y_max = max_price + price_range * 0.2
            
            # Doğrudan matplotlib plot kullan
            self.ax.plot(dates, prices, label='Gerçek Veriler', linewidth=2, color='blue')
            
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            # Legend'ı sol üste al, grafiğin dışına taşır
            self.ax.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', fontsize=8)
            
            # Y ekseni limitlerini ayarla
            self.ax.set_ylim(y_min, y_max)
            
            # Y ekseni formatı
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # X ekseni tarih formatı
            self.ax.xaxis.set_major_locator(YearLocator())
            self.ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            
            self.canvas.draw()
            
        except Exception as e:
            error_msg = f"Grafik çizim hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.warning(self, "Uyarı", error_msg)

    def make_prediction(self):
        try:
            # Model kontrolü
            if not hasattr(self, 'lstm_model') or self.lstm_model is None:
                QMessageBox.warning(self, "Uyarı", "Model henüz eğitilmedi, lütfen bekleyin!")
                return
            
            if not self.date_input.text():
                QMessageBox.warning(self, "Uyarı", "Lütfen bir tarih girin!")
                return

            end_date = self.date_input.text()
            
            try:
                target_date = datetime.strptime(end_date, '%Y-%m-%d')
            except ValueError:
                QMessageBox.warning(self, "Uyarı", "Geçersiz tarih formatı! YYYY-MM-DD formatında giriniz.")
                return
                
            # Tahmin yap
            predictions_df = self.predict_next_days(end_date)
            
            if predictions_df.empty:
                QMessageBox.warning(self, "Uyarı", "Tahmin oluşturulamadı!")
                return

            # Grafiği güncelle
            self.ax.clear()
            
            # Tarihleri datetime dizisi olarak al
            real_dates = self.df.index.to_pydatetime()
            pred_dates = predictions_df['Tarih'].values
            
            # Fiyatları numpy array'e dönüştür
            real_prices = self.df['Close'].values
            pred_prices = predictions_df['Tahmin'].values
            
            # Veri aralığını belirle
            min_price = min(np.min(real_prices), np.min(pred_prices))
            max_price = max(np.max(real_prices), np.max(pred_prices))
            price_range = max_price - min_price
            
            # Y eksen limitlerini ayarla - %20 margin ekle
            y_min = max(0, min_price - price_range * 0.2)
            y_max = max_price + price_range * 0.2
            
            # Gerçek verileri ve tahminleri çiz
            self.ax.plot(real_dates, real_prices, 
                        label='Gerçek Veriler', linewidth=2, color='blue')
            self.ax.plot(pred_dates, pred_prices,
                        label='Tahminler', linestyle='--', 
                        linewidth=2, color='red')
                
            # Grafik özelliklerini ayarla
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            # Legend'ı sol üste al, grafiğin dışına taşır
            self.ax.legend(bbox_to_anchor=(0.01, 0.99), loc='upper left', fontsize=8)
            
            # Y ekseni limitlerini ayarla
            self.ax.set_ylim(y_min, y_max)
            
            # Y ekseni formatı
            self.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # X ekseni tarih formatı
            self.ax.xaxis.set_major_locator(YearLocator())
            self.ax.xaxis.set_major_formatter(DateFormatter('%Y'))
            
            # Grafiği yenile
            self.canvas.draw()
            
            # Sonuçları göster
            self.results_text.clear()
            self.results_text.append("Tahmin Sonuçları:\n\n")
            for _, row in predictions_df.iterrows():
                self.results_text.append(
                    f"Tarih: {row['Tarih'].strftime('%Y-%m-%d')}\n"
                    f"Tahmin: ${row['Tahmin']:,.2f}\n"
                    f"Trend: {row.get('Trend', 'N/A')}\n"
                    f"Güven: %{row.get('Güven', 0)*100:.1f}\n\n"
                )

        except Exception as e:
            error_msg = f"Tahmin hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Hata", error_msg)

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
        Daha gerçekçi tahmin fonksiyonu - güçlü ralliler ve uzun birikim dönemleriyle
        """
        try:
            predictions = []
            dates = []
            trends = []
            confidences = []
            
            # Başlangıç değerlerini ayarla
            current_price = float(self.df['Close'].iloc[-1])
            initial_price = current_price
            historical_high = float(self.df['Close'].max())
            historical_volatility = float(self.df['Close'].pct_change().std())
            
            # Zaman içinde azalan ralli gücü için parametreler
            years_passed = (datetime.strptime(last_real_date.strftime('%Y-%m-%d'), '%Y-%m-%d') - 
                        datetime(2009, 1, 3)).days / 365.25
            
            # Ralli gücü zamanla azalır (logaritmik azalma)
            def get_rally_strength(years):
                base_strength = 4.0  # Başlangıç ralli gücü (4x)
                min_strength = 1.5   # Minimum ralli gücü (1.5x)
                decay_rate = 0.12    # Azalma hızı (yavaşlatıldı)
                
                strength = base_strength * np.exp(-decay_rate * years)
                return max(strength, min_strength)
            
            # Döngü parametreleri - uzun birikim dönemleri ve güçlü rallilerle
            def get_cycle_phases(years):
                rally_strength = get_rally_strength(years)
                
                return {
                    'bull': {
                        'duration': (120, 180),  # 4-6 ay
                        'total_growth': (1.5, 3.0),  # Daha güçlü ralliler
                        'volatility': (0.02, 0.04),
                        'breakout_chance': 0.45,  # %45 zirve kırma şansı
                        'max_breakout': 3.0       # Maksimum 3x (200% üzeri)
                    },
                    'bear': {
                        'duration': (90, 180),    # 3-6 ay
                        'max_decline': 0.6,       # Maksimum %60 düşüş
                        'min_decline': 0.2,       # Minimum %20 düşüş
                        'volatility': (0.03, 0.05)
                    },
                    'accumulation': {
                        'duration': (180, 1080),  # 6 ay - 3 yıl
                        'range': 0.15,            # %15 fiyat aralığı
                        'volatility': (0.01, 0.02),
                        'mini_rally_chance': 0.2  # %20 mini ralli şansı
                    }
                }
            
            # Zirve takibi
            all_time_high = max(historical_high, current_price)
            peaks_count = 0
            failed_breakout_count = 0
            
            # Başlangıç fazı
            current_phase = 'accumulation'
            days_in_phase = 0
            cycle_phases = get_cycle_phases(years_passed)
            phase_duration = np.random.randint(*cycle_phases[current_phase]['duration'])
            phase_target = current_price
            
            # Mini ralli takibi
            mini_rally_active = False
            mini_rally_duration = 0
            
            for i in range(days_to_predict):
                current_date = last_real_date + timedelta(days=i+1)
                years_from_start = years_passed + (i / 365.25)
                
                # Güncel döngü parametrelerini al
                cycle_phases = get_cycle_phases(years_from_start)
                
                # Faz değişimi kontrolü
                if days_in_phase >= phase_duration:
                    days_in_phase = 0
                    mini_rally_active = False
                    
                    # Faz geçişleri
                    if current_phase == 'accumulation':
                        # Birikim fazında mini ralli kontrolü
                        if (not mini_rally_active and 
                            np.random.random() < cycle_phases['accumulation']['mini_rally_chance']):
                            mini_rally_active = True
                            mini_rally_duration = np.random.randint(14, 45)  # 2-6 haftalık mini ralli
                            phase_target = current_price * np.random.uniform(1.2, 1.5)
                            continue
                            
                        current_phase = 'bull'
                        # Ana ralli hedefi
                        growth_multiple = np.random.uniform(*cycle_phases['bull']['total_growth'])
                        phase_target = current_price * growth_multiple
                        
                        # Zirve kırma mantığı
                        if np.random.random() < cycle_phases['bull']['breakout_chance']:
                            breakout_multiple = np.random.uniform(1.5, cycle_phases['bull']['max_breakout'])
                            phase_target = max(phase_target, all_time_high * breakout_multiple)
                            
                    elif current_phase == 'bull':
                        current_phase = 'bear'
                        decline = np.random.uniform(
                            cycle_phases['bear']['min_decline'],
                            cycle_phases['bear']['max_decline']
                        )
                        phase_target = current_price * (1 - decline)
                        
                        # Minimum düşüş sınırı - son zirvenin beşte biri
                        min_target = all_time_high * 0.2
                        phase_target = max(phase_target, min_target)
                        
                        if current_price > all_time_high:
                            all_time_high = current_price
                            peaks_count += 1
                        else:
                            failed_breakout_count += 1
                            
                    else:  # bear -> accumulation
                        current_phase = 'accumulation'
                        # Uzun birikim dönemi
                        phase_duration = np.random.randint(*cycle_phases['accumulation']['duration'])
                        range_size = cycle_phases['accumulation']['range']
                        phase_target = current_price * (1 + np.random.uniform(-range_size/2, range_size/2))
                    
                    if not mini_rally_active:
                        phase_duration = np.random.randint(*cycle_phases[current_phase]['duration'])
                
                # Mini ralli kontrolü
                if mini_rally_active:
                    if mini_rally_duration <= 0:
                        mini_rally_active = False
                        phase_target = current_price * 0.8  # Mini ralli sonrası düzeltme
                    mini_rally_duration -= 1
                
                # Günlük değişim hesaplama
                phase_volatility = (cycle_phases['accumulation']['volatility'] if mini_rally_active 
                                else cycle_phases[current_phase]['volatility'])
                volatility = np.random.uniform(*phase_volatility)
                
                if current_phase == 'bull' or current_phase == 'bear' or mini_rally_active:
                    # Trend yönünde hareket
                    target_return = (phase_target / current_price) ** (1 / max(1, (phase_duration - days_in_phase)))
                    expected_change = target_return - 1
                else:
                    # Yatay hareket
                    expected_change = (phase_target - current_price) / max(1, (phase_duration - days_in_phase))
                    expected_change = expected_change / current_price
                
                # Rastgele değişim ekle
                price_change = expected_change + np.random.normal(0, volatility)
                
                # Yeni fiyat
                current_price *= (1 + price_change)
                
                # Fiyat sınırları
                current_price = max(current_price, all_time_high * 0.2)  # En düşük zirvenin %20'si
                
                # Sonuçları kaydet
                predictions.append(float(current_price))
                dates.append(current_date)
                trends.append('mini_rally' if mini_rally_active else current_phase)
                confidences.append(max(0.3, 1.0 - (i / days_to_predict) * 0.5))
                
                days_in_phase += 1
            
            return pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions,
                'Trend': trends,
                'Güven': confidences
            })
            
        except Exception as e:
            print(f"Tahmin hatası: {str(e)}")
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
            # Tarih tiplerini eşitle
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
            
            # UTC bilgisini kaldır
            if hasattr(self.df.index, 'tz'):
                index_no_tz = self.df.index.tz_localize(None)
            else:
                index_no_tz = self.df.index
                
            # Hedef tarihe en yakın tarihi bul
            closest_date = index_no_tz[index_no_tz.get_indexer([target_date], method='nearest')[0]]
            
            # Gerçek değeri al
            actual_price = self.df.loc[closest_date.tz_localize(self.df.index.tz), 'Close']
            
            # DataFrame oluştur
            return pd.DataFrame({
                'Tarih': [closest_date],
                'Tahmin': [float(actual_price)]  # float'a çevirerek tip uyumsuzluğunu önle
            })
            
        except Exception as e:
            print(f"Geçmiş tarih analizi hatası: {str(e)}")
            raise e  # Hatayı yukarı ilet     

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
    app.setStyle('Fusion')  # Windows'ta daha iyi görünüm için
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())









    