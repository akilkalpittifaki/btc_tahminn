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
from PyQt5.QtCore import Qt, QEvent, QThread, pyqtSignal, QDate

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QMessageBox, QDateEdit,
    QTextEdit, QSplitter, QDialog, QRadioButton
)
# Matplotlib - Grafik çizim araçları
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.dates import YearLocator, DateFormatter

# Kendi modülümüz
from model_evaluator2 import ModelEvaluator2

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bitcoin_predictor.log'),
        logging.StreamHandler()
    ]
)




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

        # Tahmin modu seçimi için radio button grubu
        self.mode_group = QWidget()
        self.mode_layout = QHBoxLayout(self.mode_group)

        self.deterministic_radio = QRadioButton("Deterministik Tahmin")
        self.stochastic_radio = QRadioButton("Stokastik Tahmin")
        self.deterministic_radio.setChecked(True)  # Varsayılan olarak deterministik

        self.mode_layout.addWidget(self.deterministic_radio)
        self.mode_layout.addWidget(self.stochastic_radio)

        # Radio butonların stilini ayarla
        for radio in [self.deterministic_radio, self.stochastic_radio]:
            radio.setStyleSheet("""
                QRadioButton {
                    font-size: 12pt;
                    color: #2C3E50;
                }
                QRadioButton::indicator {
                    width: 15px;
                    height: 15px;
                }
            """)

        # Info butonu
        self.info_button = QPushButton("ℹ️")
        self.info_button.setFixedSize(30, 30)
        self.info_button.setStyleSheet("""
            QPushButton {
                font-size: 16pt;
                border-radius: 15px;
                background-color: #3498DB;
                color: white;
            }
        """)
        self.info_button.clicked.connect(self.show_prediction_info)
        self.mode_layout.addWidget(self.info_button)

        # Mode group'u layout'a ekle
        date_layout.addWidget(self.mode_group)
        
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

        self.evaluate_button.setStyleSheet("""
            QPushButton {
                font-size: 14pt;
                font-weight: bold;
                padding: 10px;
                background-color: #3498DB;
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


    def evaluate_model_performance(self):
        """Model performans değerlendirmesi"""
        try:
            # Model eğitilmemiş ise uyar
            if not hasattr(self, 'lstm_model') or self.lstm_model is None:
                QMessageBox.warning(self, "Uyarı", "Lütfen önce modeli eğitin!")
                return

            # Yeni analiz dialogunu göster
            dialog = AnalysisDialog(self)
            dialog.exec_()

        except Exception as e:
            error_msg = f"Model değerlendirme hatası: {str(e)}"
            logging.error(error_msg)
            QMessageBox.critical(self, "Hata", error_msg)  


    def show_prediction_info(self):
        QMessageBox.information(self, "Tahmin Modları Hakkında",
            "İki tahmin modu arasında seçim yapabilirsiniz:\n\n"
            "1. Deterministik Tahmin:\n"
            "   - Her seferinde aynı sonucu verir\n"
            "   - Daha tutarlı ve tekrarlanabilir\n"
            "   - Akademik değerlendirme için uygundur\n\n"
            "2. Stokastik Tahmin:\n"
            "   - Her tahminde farklı senaryolar üretir\n"
            "   - Piyasa belirsizliklerini daha iyi yansıtır\n"
            "   - Risk analizi için uygundur")                
        
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
        
        # Momentum göstergeleri - düzeltilmiş versiyon
        self.momentum_short = float(self.df['Close'].pct_change(30).mean())  # 30 günlük momentum
        self.momentum_long = float(self.df['Close'].pct_change(90).mean())   # 90 günlük momentum
        
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
                'Momentum_Short': float(self.momentum_short),  # Direkt float değeri kullan
                'Momentum_Long': float(self.momentum_long),
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
        """Seçilen moda göre tahmin yap"""
        try:
            if self.deterministic_radio.isChecked():
                return self.predict_deterministic(days_to_predict, last_real_date)
            else:
                # Stokastik tahmin seçildiğinde
                predictions = []
                dates = []
                trends = []
                confidences = []
                
                # Başlangıç değerlerini ayarla
                current_price = float(self.df['Close'].iloc[-1])
                historical_high = float(self.df['Close'].iloc[-1])  # Düzeltildi
                initial_price = current_price
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
            logging.error(f"Tahmin hatası: {str(e)}")
            return pd.DataFrame()
        
    def predict_deterministic(self, days_to_predict, last_real_date):
        """Deterministik tahmin - her seferinde aynı sonucu üretir"""
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
            
            # Sabit değerler tanımla
            BASE_VOLATILITY = 0.02
            BULL_GROWTH = 1.8  # Sabit büyüme oranı
            BEAR_DECLINE = 0.3  # Sabit düşüş oranı
            PHASE_DURATIONS = {
                'bull': 150,    # Sabit 5 ay
                'bear': 120,    # Sabit 4 ay
                'accumulation': 240  # Sabit 8 ay
            }
            
            # Halving cycle etkisi (4 yıllık) - sabit eğilimler için
            halving_date = datetime(2024, 4, 1)
            days_since_halving = (last_real_date - halving_date).days
            cycle_position = (days_since_halving % (4 * 365)) / (4 * 365)
            
            # Başlangıç fazını belirle (cycle_position'a göre)
            if cycle_position < 0.3:
                current_phase = 'accumulation'
                days_in_phase = int(cycle_position * PHASE_DURATIONS['accumulation'])
            elif cycle_position < 0.7:
                current_phase = 'bull'
                days_in_phase = int((cycle_position - 0.3) * PHASE_DURATIONS['bull'])
            else:
                current_phase = 'bear'
                days_in_phase = int((cycle_position - 0.7) * PHASE_DURATIONS['bear'])
            
            phase_duration = PHASE_DURATIONS[current_phase]
            
            # Sabit trend momentumu belirle
            trend_momentum = np.sin(2 * np.pi * cycle_position)
            
            for i in range(days_to_predict):
                current_date = last_real_date + timedelta(days=i+1)
                
                # Faz değişimi kontrolü
                if days_in_phase >= phase_duration:
                    days_in_phase = 0
                    
                    # Deterministik faz geçişleri
                    if current_phase == 'accumulation':
                        current_phase = 'bull'
                        growth_target = BULL_GROWTH * (1 + 0.1 * trend_momentum)
                        phase_target = current_price * growth_target
                        phase_duration = PHASE_DURATIONS['bull']
                        
                    elif current_phase == 'bull':
                        current_phase = 'bear'
                        decline_target = BEAR_DECLINE * (1 - 0.1 * trend_momentum)
                        phase_target = current_price * (1 - decline_target)
                        phase_duration = PHASE_DURATIONS['bear']
                        
                        if current_price > historical_high:
                            historical_high = current_price
                            
                    else:  # bear -> accumulation
                        current_phase = 'accumulation'
                        phase_duration = PHASE_DURATIONS['accumulation']
                        # Birikim fazında yatay hareket
                        phase_target = current_price * (1 + 0.05 * trend_momentum)
                
                # Günlük değişim hesapla
                days_remaining = phase_duration - days_in_phase
                if days_remaining <= 0:
                    days_remaining = 1
                    
                if current_phase == 'bull':
                    # Yükseliş fazında üstel büyüme
                    daily_return = (phase_target / current_price) ** (1 / days_remaining)
                    current_price *= daily_return
                elif current_phase == 'bear':
                    # Düşüş fazında logaritmik düşüş
                    daily_return = (phase_target / current_price) ** (1 / days_remaining)
                    current_price *= daily_return
                else:  # accumulation
                    # Yatay hareket - küçük dalgalanmalar
                    deviation = 0.001 * np.sin(days_in_phase * (2 * np.pi / 30))  # 30 günlük mini döngü
                    current_price *= (1 + deviation)
                
                # Fiyat sınırları
                min_price = historical_high * 0.2  # En düşük fiyat son zirvenin %20'si
                current_price = max(current_price, min_price)
                
                # Sonuçları kaydet
                predictions.append(float(current_price))
                dates.append(current_date)
                trends.append(current_phase)
                
                # Güven skoru - zamanla azalan ama daha yavaş
                base_confidence = 0.9
                time_decay = (i / days_to_predict) * 0.4  # 0.4 ile çarparak daha yavaş azalma
                confidences.append(float(base_confidence - time_decay))
                
                days_in_phase += 1
            
            return pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions,
                'Trend': trends,
                'Güven': confidences
            })
            
        except Exception as e:
            logging.error(f"Deterministik tahmin hatası: {str(e)}")
            return pd.DataFrame()

    def predict_stochastic(self, days_to_predict, last_real_date):
        """
        Daha gerçekçi stokastik tahmin fonksiyonu - güçlü ralliler ve uzun birikim dönemleriyle
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
            
        except Exception as e:
            logging.error(f"Deterministik tahmin hatası: {str(e)}")
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
     


class AnalysisDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle("Tahmin Değerlendirmesi")
        self.setGeometry(200, 200, 800, 600)
        
        # Ana layout
        layout = QVBoxLayout()
        
        # Tarih seçimi için widget
        date_widget = QWidget()
        date_layout = QHBoxLayout(date_widget)
        
        # Tahmin yapılan gün seçimi
        pred_date_widget = QWidget()
        pred_date_layout = QVBoxLayout(pred_date_widget)
        pred_date_label = QLabel("Tahmin Yapılan Gün:")
        pred_date_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.prediction_date = QDateEdit()
        self.prediction_date.setCalendarPopup(True)
        self.prediction_date.setDisplayFormat("dd.MM.yyyy")  # Türkçe format
        self.prediction_date.setDate(QDate.currentDate().addDays(-30))
        self.prediction_date.setStyleSheet("""
            QDateEdit {
                font-size: 12pt;
                padding: 5px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
            }
        """)
        pred_date_layout.addWidget(pred_date_label)
        pred_date_layout.addWidget(self.prediction_date)
        
        # Hedef gün seçimi
        target_date_widget = QWidget()
        target_date_layout = QVBoxLayout(target_date_widget)
        target_date_label = QLabel("Hedef Gün:")
        target_date_label.setStyleSheet("font-size: 12pt; font-weight: bold;")
        self.target_date = QDateEdit()
        self.target_date.setCalendarPopup(True)
        self.target_date.setDisplayFormat("dd.MM.yyyy")  # Türkçe format
        self.target_date.setDate(QDate.currentDate())
        self.target_date.setStyleSheet("""
            QDateEdit {
                font-size: 12pt;
                padding: 5px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
            }
        """)
        target_date_layout.addWidget(target_date_label)
        target_date_layout.addWidget(self.target_date)
        
        # Tarihleri layout'a ekle
        date_layout.addWidget(pred_date_widget)
        date_layout.addWidget(target_date_widget)
        layout.addWidget(date_widget)
        
        # Değerlendirme butonu
        self.evaluate_button = QPushButton("Değerlendir")
        self.evaluate_button.clicked.connect(self.evaluate_prediction)
        self.evaluate_button.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: bold;
                padding: 10px;
                background-color: #2ECC71;
                color: white;
                border: none;
                border-radius: 5px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #27AE60;
            }
        """)
        layout.addWidget(self.evaluate_button)
        
        # Sonuçlar için text alanı
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setStyleSheet("""
            QTextEdit {
                font-size: 12pt;
                padding: 10px;
                border: 2px solid #BDC3C7;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.results_text)
        
        self.setLayout(layout)
    
    def evaluate_prediction(self):
        try:
            # Tarihleri pandas Timestamp'e çevir
            pred_date = pd.Timestamp(self.prediction_date.date().toPyDate())
            target_date = pd.Timestamp(self.target_date.date().toPyDate())
            
            # Tarih kontrolü
            if pred_date >= target_date:
                QMessageBox.warning(self, "Uyarı", 
                    "Tahmin tarihi, hedef tarihten önce olmalıdır!")
                return
            
            # Minimum 1 gün fark kontrolü
            if (target_date - pred_date).days < 1:
                QMessageBox.warning(self, "Uyarı",
                    "Tahmin ve hedef tarih arasında en az 1 gün olmalıdır!")
                return
            
            # DataFrame'in timezone'ını kontrol et
            if hasattr(self.parent.df.index, 'tz'):
                df_tz = self.parent.df.index.tz
                if df_tz is not None:
                    pred_date = pred_date.tz_localize(df_tz)
                    target_date = target_date.tz_localize(df_tz)
            
            # Değerlendirmeyi yap
            evaluator = ModelEvaluator2(self.parent)
            results = evaluator.evaluate_specific_prediction(pred_date, target_date)
            
            # Renk kodları
            success_color = "#27AE60"  # Yeşil
            error_color = "#E74C3C"    # Kırmızı
            
            # Başarı oranına göre renk belirleme
            success_rate = 100 - results['error_percentage']
            if success_rate >= 95:
                rate_color = "#27AE60"  # Yeşil
            elif success_rate >= 85:
                rate_color = "#F1C40F"  # Sarı
            else:
                rate_color = "#E74C3C"  # Kırmızı
            
            # Sonuçları HTML formatında hazırla
            report = f"""
            <h2 style='color: #2C3E50;'>Tahmin Değerlendirme Raporu</h2>
            <p style='font-size: 14px;'>
            <b>Tahmin Yapılan Gün:</b> {results['prediction_date'].strftime('%d.%m.%Y')}<br>
            <b>Hedef Gün:</b> {results['target_date'].strftime('%d.%m.%Y')}<br>
            <b>Tahmin Süresi:</b> {results['days_difference']} gün<br><br>
            
            <b>Gerçekleşen Fiyat:</b> <span style='color: {success_color};'>${results['actual_price']:,.2f}</span><br>
            <b>Tahmin Edilen Fiyat:</b> <span style='color: {error_color};'>${results['predicted_price']:,.2f}</span><br><br>
            
            <b>Mutlak Hata:</b> ${results['error']:,.2f}<br>
            <b>Hata Oranı:</b> %{results['error_percentage']:.2f}<br><br>
            
            <b>Başarı Oranı:</b> <span style='color: {rate_color}; font-size: 16px;'>%{success_rate:.2f}</span>
            </p>
            """
            
            self.results_text.setHtml(report)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", 
                f"Değerlendirme yapılırken bir hata oluştu:\n{str(e)}\n\n"
                "Lütfen tarih aralığını kontrol edin ve tekrar deneyin.")    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Windows'ta daha iyi görünüm için
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())









    