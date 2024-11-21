import xgboost as xgb
from xgboost import XGBRegressor
import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Importing plotly failed. Interactive plots will not work.")


class AnalysisDialog(QDialog):
    def analyze_past_date(self, target_date):
        """Geçmiş tarihteki verileri analiz et"""
        try:
            # Parent'tan df'yi al
            df = self.parent.df if self.parent else None
            if df is None:
                raise Exception("Veri çerçevesi bulunamadı")

            # Hedef tarihe en yakın tarihi bul
            closest_date = min(df.index, key=lambda x: abs(x.tz_localize(None) - target_date))
            actual_price = df.loc[closest_date, 'Close']

            # Sonuçları DataFrame'e dönüştür
            past_df = pd.DataFrame({
                'Tarih': [closest_date],
                'Tahmin': [float(actual_price)]  # Explicitly convert to float
            })

            # Analiz metni
            analysis_text = f"""Bitcoin Geçmiş Tarih Analizi

    Sorguladığınız Tarih: {target_date.strftime('%Y-%m-%d')}
    En Yakın Veri Tarihi: {closest_date.strftime('%Y-%m-%d')}
    Bitcoin Fiyatı: ${float(actual_price):.2f}

    Teknik Göstergeler:
    - RSI: {float(df.loc[closest_date, 'RSI']):.2f}
    - 7 Günlük MA: ${float(df.loc[closest_date, 'MA7']):.2f}
    - 30 Günlük MA: ${float(df.loc[closest_date, 'MA30']):.2f}
    - Volatilite: %{float(df.loc[closest_date, 'Volatility'])*100:.2f}

    Not:
    - Bu tarih geçmişe ait olduğu için gerçek veriler gösterilmektedir.
    - Tahmin yerine gerçekleşmiş fiyat ve göstergeler verilmiştir.
    """

            # Analiz penceresini göster
            dialog = AnalysisDialog(analysis_text, self)
            dialog.exec_()

            return past_df

        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Geçmiş tarih analizi hatası: {str(e)}")
            return pd.DataFrame()

    def __init__(self, analysis_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tahmin Analizi")
        self.setGeometry(200, 200, 800, 600)
        self.parent = parent  # Parent'ı kaydet
        
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
    def analyze_historical_date(self, target_date):  
        """Geçmiş tarihteki verileri analiz et"""  
        try:  
            # Hedef tarihe en yakın tarihi bul  
            closest_date = min(self.df.index, key=lambda x: abs(x.tz_localize(None) - target_date))  
            actual_price = float(self.df.loc[closest_date, 'Close'])  # Convert to float
            
            # Sonuçları DataFrame'e dönüştür  
            past_df = pd.DataFrame({  
                'Tarih': [closest_date],  
                'Gerçek_Fiyat': [actual_price]  
            })  
            
            # Analiz metni oluştur  
            analysis_text = f"""Bitcoin Geçmiş Tarih Analizi  

            Sorguladığınız Tarih: {target_date.strftime('%Y-%m-%d')}  
            En Yakın Veri Tarihi: {closest_date.strftime('%Y-%m-%d')}  
            Bitcoin Fiyatı: ${actual_price:.2f}  

            Teknik Göstergeler:  
            - RSI: {float(self.df.loc[closest_date, 'RSI']):.2f}  
            - 7 Günlük MA: ${float(self.df.loc[closest_date, 'MA7']):.2f}  
            - 30 Günlük MA: ${float(self.df.loc[closest_date, 'MA30']):.2f}  
            - Volatilite: %{float(self.df.loc[closest_date, 'Volatility'])*100:.2f}  

            Not:   
            - Bu tarih geçmişe ait olduğu için gerçek veriler gösterilmektedir.  
            - Tahmin yerine gerçekleşmiş fiyat ve göstergeler verilmiştir.  
            """  
            
            # Analiz penceresini göster  
            dialog = AnalysisDialog(analysis_text, self)  
            dialog.exec_()
            
            return past_df  
            
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Geçmiş tarih analizi hatası: {str(e)}")  
            return pd.DataFrame()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bitcoin Tahmin Uygulaması")
        self.setGeometry(100, 100, 1200, 800)
        
        # Ana widget'ları oluştur
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        
        # UI bileşenlerini ayarla
        self.setup_ui_components()
        
        # Model ağırlıkları UI'ını kur
        self.setup_model_weights_ui()
        
        # Veriyi hazırla ve modelleri eğit
        try:
            print("Uygulama başlatılıyor...")
            self.prepare_data_and_models()
            print("İlk grafik çiziliyor...")
            self.plot_bitcoin_data()
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Başlangıç hatası: {str(e)}")

    def setup_model_weights_ui(self):
        """Model ağırlıkları için UI bileşenlerini oluştur"""
        weights_widget = QWidget()
        weights_layout = QVBoxLayout()
        
        weights_label = QLabel("Model Ağırlıkları:")
        weights_layout.addWidget(weights_label)
        
        self.weight_sliders = {}
        for model in ['XGBoost', 'Prophet', 'SARIMA']:
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(100)
            slider.setValue(33)
            
            label = QLabel(f"{model}: 33%")
            slider.valueChanged.connect(lambda v, l=label, m=model: l.setText(f"{m}: {v}%"))
            
            weights_layout.addWidget(label)
            weights_layout.addWidget(slider)
            self.weight_sliders[model] = slider
            
        weights_widget.setLayout(weights_layout)
        self.main_layout.addWidget(weights_widget)  # Bu satır eklendi         

    def setup_ui_components(self):
        """UI bileşenlerini ayarla"""
        splitter = QSplitter(Qt.Horizontal)
        
        # Sol panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Grafik ayarları
        self.figure, self.ax = plt.subplots(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        left_layout.addWidget(self.canvas)
        
        # Tarih girişi
        date_widget = QWidget()
        date_layout = QVBoxLayout(date_widget)
        
        self.date_label = QLabel("Tahmin Tarihi (YYYY-MM-DD):")
        self.date_input = QLineEdit()
        self.date_input.setPlaceholderText("Örnek: 2024-12-31")
        
        self.predict_button = QPushButton("Tahmin Et")
        self.predict_button.clicked.connect(self.make_prediction)
        
        date_layout.addWidget(self.date_label)
        date_layout.addWidget(self.date_input)
        date_layout.addWidget(self.predict_button)
        
        left_layout.addWidget(date_widget)
        
        # Sağ panel (sonuçlar)
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumWidth(300)
        
        splitter.addWidget(left_widget)
        splitter.addWidget(self.results_text)
        
        self.main_layout.addWidget(splitter)

    def prepare_data_and_models(self):
        """Veri hazırlama ve model eğitimi"""
        try:
            self.ticker = "BTC-USD"
            start_date = (datetime.now() - timedelta(days=4*365)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Verileri indir ve hazırla
            print("Bitcoin verilerini indiriyor...")
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            
            print("Teknik indikatörleri hesaplıyor...")
            self.add_technical_indicators()
            
            print("Eğitim verilerini hazırlıyor...")
            self.prepare_training_data()
            
            print("Prophet için veri hazırlığı yapılıyor...")
            self.prepare_prophet_data()
            
            print("XGBoost modelini eğitiyor...")
            self.train_xgboost_model()
            
            print("Prophet modelini eğitiyor...")
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative'
            )
            self.prophet_model.fit(self.prophet_df)
            
            print("SARIMA modelini eğitiyor...")
            self.sarima_model = SARIMAX(
                self.df['Close'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            ).fit(disp=False)
            
            print("Model eğitimi tamamlandı!")
            
        except Exception as e:
            error_msg = f"Veri hazırlama ve model eğitimi hatası: {str(e)}"
            print(error_msg)
            QMessageBox.critical(self, "Hata", error_msg)
            raise e


    def prepare_prophet_data(self):
        """Prophet için veri hazırlığı"""
        # Prophet DataFrame'i oluştur ve kaydet
        self.prophet_df = pd.DataFrame({
            'ds': self.df.index,
            'y': self.df['Close']
        }).reset_index(drop=True)  # index'i sıfırla

    def prepare_training_data(self):
        """Eğitim verilerini hazırla"""
        # XGBoost için genişletilmiş özellikler
        self.xgb_features = ['Open', 'High', 'Low', 'Volume', 'Returns', 
                            'Volatility', 'MA7', 'MA30', 'MA90', 'MA200', 
                            'RSI', 'MACD', 'Signal_Line', 
                            'BB_middle', 'BB_upper', 'BB_lower']
        
        # Eksik değerleri doldur
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')
        
        # Verileri hazırla
        self.X_xgb = self.df[self.xgb_features]
        self.y = self.df['Close'].values.flatten()  # .flatten() kullanarak 1-boyutlu yap
        
        # Scaler'ı oluştur ve verileri ölçekle
        self.xgb_scaler = StandardScaler()
        self.X_xgb_scaled = self.xgb_scaler.fit_transform(self.X_xgb)

        def get_xgb_predictions(self, days_to_predict):
            """XGBoost tahminleri"""
            predictions = []
            current_data = self.df.iloc[-1:].copy()
            
            for _ in range(days_to_predict):
                # Eksik değerleri doldur
                current_data = current_data.fillna(method='ffill')
                
                # Özellikleri hazırla ve tahmin yap
                features = self.xgb_scaler.transform(current_data[self.xgb_features])
                pred = float(self.xgb_model.predict(features)[0])
                predictions.append(pred)
                
                # Teknik özellikleri güncelle
                self.update_technical_features(current_data, pred, predictions)
            
            return predictions


    def train_all_models(self):
        """Tüm modelleri eğit"""
        try:
            # XGBoost
            self.train_xgboost_model()
            
            # Prophet modeli
            self.prophet_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                seasonality_mode='multiplicative'
            )
            self.prophet_model.fit(self.prophet_df)
            
            # SARIMA modeli
            self.sarima_model = SARIMAX(
                self.df['Close'],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7)
            ).fit(disp=False)
        except Exception as e:
            raise Exception(f"Model eğitimi hatası: {str(e)}")

    def predict_future_date(self, days_to_predict, last_real_date):
        """Gelecek tarihi tahmin et"""
        try:
            predictions = []
            dates = []
            confidence_intervals = []
            
            # Her model için tahminleri al
            xgb_preds = self.get_xgb_predictions(days_to_predict)
            prophet_preds = self.get_prophet_predictions(days_to_predict)
            sarima_preds = self.get_sarima_predictions(days_to_predict)
            
            for i in range(days_to_predict):
                # Model ağırlıklarını al
                weights = {
                    'XGBoost': self.weight_sliders['XGBoost'].value() / 100,
                    'Prophet': self.weight_sliders['Prophet'].value() / 100,
                    'SARIMA': self.weight_sliders['SARIMA'].value() / 100
                }
                
                # Ağırlıkları normalize et
                total_weight = sum(weights.values())
                weights = {k: v/total_weight for k, v in weights.items()}
                
                # Ağırlıklı tahmin
                weighted_pred = (
                    xgb_preds[i] * weights['XGBoost'] +
                    prophet_preds[i] * weights['Prophet'] +
                    sarima_preds[i] * weights['SARIMA']
                )
                
                # Volatilite bazlı rastgele dalgalanma ekle
                volatility = self.df['Volatility'].iloc[-1]
                random_fluctuation = np.random.normal(0, volatility * weighted_pred * 0.1)
                final_pred = weighted_pred + random_fluctuation
                
                next_date = last_real_date + timedelta(days=i+1)
                predictions.append(final_pred)
                dates.append(next_date)
                
                # Güven aralıkları hesapla
                std_dev = np.std([xgb_preds[i], prophet_preds[i], sarima_preds[i]])
                confidence_intervals.append({
                    'lower': final_pred - 2*std_dev,
                    'upper': final_pred + 2*std_dev
                })
            
            # Sonuçları DataFrame'e dönüştür
            future_df = pd.DataFrame({
                'Tarih': dates,
                'Tahmin': predictions,
                'Alt_Sınır': [ci['lower'] for ci in confidence_intervals],
                'Üst_Sınır': [ci['upper'] for ci in confidence_intervals]
            })
            
            # Grafiği güncelle
            self.plot_predictions(future_df)
            
            return future_df
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")
            return pd.DataFrame()
        
    def plot_predictions(self, future_df):
        """Tahminleri ve güven aralıklarını çiz"""
        self.ax.clear()
        
        # Gerçek verileri çiz
        self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2)
        
        # Tahminleri çiz
        future_df.set_index('Tarih')['Tahmin'].plot(ax=self.ax, 
                                                   label='Tahminler',
                                                   style='--',
                                                   linewidth=2)
        
        # Güven aralıklarını çiz
        self.ax.fill_between(future_df['Tarih'],
                           future_df['Alt_Sınır'],
                           future_df['Üst_Sınır'],
                           alpha=0.2,
                           label='95% Güven Aralığı')
        
        self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
        self.ax.legend()
        self.ax.grid(True)
        self.canvas.draw()


    def get_prophet_predictions(self, days):
        """Prophet tahminleri"""
        future_dates = self.prophet_model.make_future_dataframe(periods=days)
        forecast = self.prophet_model.predict(future_dates)
        return forecast.tail(days)['yhat'].values

    def get_sarima_predictions(self, days):
        """SARIMA tahminleri"""
        forecast = self.sarima_model.forecast(steps=days)
        return forecast.values                

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


    def train_xgboost_model(self):
        """XGBoost modelini eğit"""
        self.xgb_model = xgb.XGBRegressor(  
            objective='reg:squarederror',  
            n_estimators=300,  # Arttırıldı  
            learning_rate=0.03,  # Düşürüldü  
            max_depth=8,  # Arttırıldı  
            min_child_weight=3,  
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
        try:  
            last_real_date = self.df.index[-1].tz_localize(None)  
            target_date = datetime.strptime(end_date, '%Y-%m-%d')  
            days_to_predict = (target_date - last_real_date).days  

            if days_to_predict <= 0:  
                return self.analyze_historical_date(target_date)  
            else:  
                return self.predict_future_date(days_to_predict, last_real_date)  
                
        except Exception as e:  
            QMessageBox.critical(self, "Hata", f"Tahmin hatası: {str(e)}")  
            return pd.DataFrame()


    def update_technical_features(self, current_data, next_price, predictions):  
        """Bir sonraki gün için teknik özellikleri güncelle"""  
        # Fiyat değişim yüzdesi  
        price_change_pct = (next_price - predictions[-1]) / predictions[-1] if predictions else 0  
        
        # Rastgele volatilite faktörü (gerçekçi dalgalanmalar için)  
        volatility_factor = np.random.normal(1, 0.02)  # %2 standart sapma  
        
        current_data['Close'] = next_price * volatility_factor  
        current_data['Open'] = next_price  
        current_data['High'] = next_price * (1 + abs(price_change_pct) * 0.5) * volatility_factor  
        current_data['Low'] = next_price * (1 - abs(price_change_pct) * 0.5) * volatility_factor  
        
        # Son n günlük tahminleri al  
        recent_predictions = (predictions[-30:] if len(predictions) > 0 else [next_price])  
        
        # Geliştirilmiş volatilite hesaplaması  
        historical_volatility = np.std(recent_predictions) if len(recent_predictions) > 1 else 0  
        implied_volatility = abs(price_change_pct) * np.sqrt(252)  # Yıllık volatilite  
        combined_volatility = (historical_volatility + implied_volatility) / 2  
        
        current_data['Returns'] = price_change_pct  
        current_data['Volatility'] = combined_volatility  
        
        # Hareketli ortalamalar  
        current_data['MA7'] = np.mean(recent_predictions[-7:]) if len(recent_predictions) >= 7 else next_price  
        current_data['MA30'] = np.mean(recent_predictions[-30:]) if len(recent_predictions) >= 30 else next_price  
        current_data['MA90'] = np.mean(recent_predictions[-90:]) if len(recent_predictions) >= 90 else next_price  
        current_data['MA200'] = np.mean(recent_predictions[-200:]) if len(recent_predictions) >= 200 else next_price  
        
        # RSI hesaplama  
        if len(predictions) >= 14:  
            recent_changes = np.diff(recent_predictions[-15:])  
            gains = np.mean([x for x in recent_changes if x > 0] or [0])  
            losses = abs(np.mean([x for x in recent_changes if x < 0] or [0]))  
            rs = gains / losses if losses != 0 else 1  
            current_data['RSI'] = 100 - (100 / (1 + rs))  
        else:  
            current_data['RSI'] = 50  
        
        # Bollinger Bands  
        if len(recent_predictions) >= 20:  
            ma20 = np.mean(recent_predictions[-20:])  
            std20 = np.std(recent_predictions[-20:])  
            current_data['BB_middle'] = ma20  
            current_data['BB_upper'] = ma20 + (std20 * 2)  
            current_data['BB_lower'] = ma20 - (std20 * 2)  
        else:  
            current_data['BB_middle'] = next_price  
            current_data['BB_upper'] = next_price * 1.02  
            current_data['BB_lower'] = next_price * 0.98

  

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())
