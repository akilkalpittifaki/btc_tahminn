import sys
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QPushButton, QLabel, QLineEdit, QMessageBox,
                           QTextEdit, QSplitter, QDialog)
from PyQt5.QtCore import Qt

class AnalysisDialog(QDialog):
    def __init__(self, analysis_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tahmin Analizi")
        self.setGeometry(200, 200, 600, 400)
        
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
        self.setGeometry(100, 100, 1000, 800)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        splitter = QSplitter(Qt.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.figure, self.ax = plt.subplots(figsize=(8, 6))
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
        
        self.prepare_data_and_model()
        self.plot_bitcoin_data()

    def prepare_data_and_model(self):
        self.ticker = "BTC-USD"
        start_date = "2024-04-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        try:
            self.df = yf.download(self.ticker, start=start_date, end=end_date)
            self.df.loc[:, 'preds'] = self.df['Close'].shift(-1)
            self.df.dropna(inplace=True)
            
            X = self.df[['Open', 'High', 'Low', 'Volume']]
            y = self.df['preds']
            
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            self.model = LinearRegression()
            self.model.fit(X_scaled, y)
            
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Veri hazırlama hatası: {str(e)}")

    def plot_bitcoin_data(self):
        try:
            self.ax.clear()
            self.df['Close'].plot(ax=self.ax, label='Gerçek Veriler', linewidth=2)
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği')
            self.ax.set_xlabel('Tarih')
            self.ax.set_ylabel('Fiyat (USD)')
            self.ax.grid(True)
            self.ax.legend()
            self.canvas.draw()
        except Exception as e:
            QMessageBox.warning(self, "Uyarı", f"Grafik çizim hatası: {str(e)}")

    def calculate_accuracy_metrics(self, actual, predicted):
        # Veriyi 1-boyutlu diziye çevir
        actual = np.ravel(actual)
        predicted = np.ravel(predicted)
        
        # Hesaplamaları yap
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        accuracy = 100 - mape
        
        # Günlük karşılaştırma detaylarını oluştur
        comparison_details = []
        
        for i in range(len(actual)):
            act = actual[i]
            pred = predicted[i]
            diff = act - pred
            daily_accuracy = (1 - abs(diff / act)) * 100
            
            date_str = (datetime.now() - timedelta(days=len(actual)-i)).strftime('%Y-%m-%d')
            
            comparison_details.append(
                f"Tarih: {date_str}\n"
                f"Gerçekleşen Fiyat: ${act:.2f}\n"
                f"Tahmin Edilen Fiyat: ${pred:.2f}\n"
                f"Fark: ${diff:.2f}\n"
                f"Günlük Doğruluk: %{daily_accuracy:.2f}\n"
                f"{'='*50}\n"
            )
        
        detailed_comparison = "\n".join(comparison_details)
        
        analysis_text = f"""Tahmin Analizi Detayları:

    1. Genel Doğruluk Oranı: {accuracy:.2f}%
    2. Ortalama Mutlak Yüzde Hata (MAPE): {mape:.2f}%
    3. Kök Ortalama Kare Hata (RMSE): ${rmse:.2f}
    4. Toplam Tahmin Sayısı: {len(actual)}

    Tahmin Başarısı Dağılımı:
    - Çok İyi Tahminler (Hata < 2%): {sum(np.abs((actual - predicted) / actual) < 0.02)} adet
    - İyi Tahminler (Hata < 5%): {sum(np.abs((actual - predicted) / actual) < 0.05)} adet
    - Orta Tahminler (Hata < 10%): {sum(np.abs((actual - predicted) / actual) < 0.10)} adet
    - Zayıf Tahminler (Hata >= 10%): {sum(np.abs((actual - predicted) / actual) >= 0.10)} adet

    Günlük Tahmin Detayları:
    {detailed_comparison}

    Tahmin Modeli Hakkında:
    - Model tipi: Lineer Regresyon
    - Kullanılan özellikler: Açılış Fiyatı, En Yüksek, En Düşük, İşlem Hacmi
    - Eğitim verisi: Son 6 aylık Bitcoin fiyat verileri

    Not: Kripto para piyasaları yüksek volatiliteye sahiptir. 
    Bu tahminler sadece geçmiş verilere dayalı istatistiksel bir analiz sunar."""
        
        return accuracy, analysis_text

    def predict_next_days(self, end_date):
        last_real_date = self.df.index[-1].tz_localize(None)
        target_date = datetime.strptime(end_date, '%Y-%m-%d')
        days_to_predict = (target_date - last_real_date).days

        if days_to_predict <= 0:
            # Geçmiş tarih analizi
            target_date_str = target_date.strftime('%Y-%m-%d')
            analysis_start = (target_date - timedelta(days=30)).strftime('%Y-%m-%d')
            analysis_end = (target_date + timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                df_analysis = yf.download(self.ticker, start=analysis_start, end=analysis_end)
                if df_analysis.empty:
                    raise ValueError("Analiz için veri bulunamadı")
                
                # Her gün için tahmin yap
                predictions = []
                actuals = []
                dates = []
                
                # Veri çerçevesini numpy dizisine çevir
                df_values = df_analysis[['Open', 'High', 'Low', 'Volume']].values
                actual_prices = df_analysis['Close'].values
                
                for i in range(len(df_values)-1):
                    current_data = df_values[i:i+1]
                    next_price = actual_prices[i+1]
                    current_date = df_analysis.index[i+1]
                    
                    current_scaled = self.scaler.transform(current_data)
                    predicted_price = float(self.model.predict(current_scaled)[0])
                    
                    predictions.append(predicted_price)
                    actuals.append(float(next_price))
                    dates.append(current_date)

                # NumPy dizilerine çevir
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                # Metrikleri hesapla
                diffs = actuals - predictions
                abs_diffs = np.abs(diffs)
                daily_accuracies = (1 - abs_diffs / actuals) * 100
                mape = np.mean(abs_diffs / actuals) * 100
                rmse = np.sqrt(np.mean(diffs ** 2))
                accuracy = 100 - mape

                # Detaylı analiz metni oluştur
                comparison_details = []
                for i, (date, actual, pred, acc) in enumerate(zip(dates, actuals, predictions, daily_accuracies)):
                    comparison_details.append(
                        f"Tarih: {date.strftime('%Y-%m-%d')}\n"
                        f"Gerçekleşen Fiyat: ${actual:.2f}\n"
                        f"Tahmin Edilen Fiyat: ${pred:.2f}\n"
                        f"Fark: ${(actual - pred):.2f}\n"
                        f"Günlük Doğruluk: %{acc:.2f}\n"
                        f"{'='*50}"
                    )

                analysis_text = f"""Tahmin Analizi Detayları:

    1. Genel Doğruluk Oranı: {accuracy:.2f}%
    2. Ortalama Mutlak Yüzde Hata (MAPE): {mape:.2f}%
    3. Kök Ortalama Kare Hata (RMSE): ${rmse:.2f}
    4. Toplam Tahmin Sayısı: {len(actuals)}

    Tahmin Başarısı Dağılımı:
    - Çok İyi Tahminler (Hata < 2%): {sum(abs_diffs / actuals < 0.02)} adet
    - İyi Tahminler (Hata < 5%): {sum(abs_diffs / actuals < 0.05)} adet
    - Orta Tahminler (Hata < 10%): {sum(abs_diffs / actuals < 0.10)} adet
    - Zayıf Tahminler (Hata >= 10%): {sum(abs_diffs / actuals >= 0.10)} adet

    Son 30 Günlük Tahmin Detayları:
    {chr(10).join(comparison_details)}

    Tahmin Modeli Hakkında:
    - Model tipi: Lineer Regresyon
    - Kullanılan özellikler: Açılış Fiyatı, En Yüksek, En Düşük, İşlem Hacmi
    - Analiz dönemi: {analysis_start} - {analysis_end}
    - Tahmin edilen gün: {target_date_str}

    Tahmin Yöntemi:
    1. Son 30 günlük veri analiz edildi
    2. Her gün için bir önceki günün verileri kullanılarak tahmin yapıldı
    3. Tahminler gerçek değerlerle karşılaştırıldı
    4. Genel başarı metrikleri hesaplandı

    Not: Kripto para piyasaları yüksek volatiliteye sahiptir. 
    Bu tahminler sadece geçmiş verilere dayalı istatistiksel bir analiz sunar."""

                # Grafiği güncelle
                self.ax.clear()
                self.ax.plot(dates, actuals, label='Gerçek Veriler', linewidth=2)
                self.ax.plot(dates, predictions, '--', 
                            label=f'Model Tahminleri (Doğruluk: {accuracy:.2f}%)', linewidth=2)
                self.ax.set_title('Bitcoin (BTC) Geçmiş Fiyat Analizi')
                self.ax.legend()
                self.ax.grid(True)
                self.canvas.draw()

                dialog = AnalysisDialog(analysis_text, self)
                dialog.exec_()

                return pd.DataFrame()

            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Geçmiş veri tahmini hatası: {str(e)}")
                return pd.DataFrame()
        else:
            # Gelecek tahminleri
            try:
                predictions = []
                dates = []
                current_features = self.df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values
                last_price = float(self.df['Close'].iloc[-1])  # float'a çevir

                for i in range(days_to_predict):
                    if current_features is None or len(current_features) == 0:
                        raise ValueError("Tahmin için gerekli veriler eksik")
                        
                    current_scaled = self.scaler.transform(current_features)
                    next_price = float(self.model.predict(current_scaled)[0])  # float'a çevir
                    
                    next_date = last_real_date + timedelta(days=i+1)
                    predictions.append(next_price)
                    dates.append(next_date)
                    
                    # Bir sonraki gün için özellikleri güncelle
                    current_features = np.array([[
                        next_price,  # Open
                        next_price * 1.02,  # High
                        next_price * 0.98,  # Low
                        float(current_features[0][3])  # Volume
                    ]])

                if not predictions or not dates:
                    raise ValueError("Tahmin sonuçları oluşturulamadı")

                # DataFrame oluştur
                future_df = pd.DataFrame({
                    'Tarih': dates,
                    'Tahmin': predictions
                })

                # Gelecek tahminleri için detaylı analiz metni
                future_analysis_text = f"""Gelecek Tahmin Analizi:

    1. Tahmin Edilen Gün Sayısı: {len(predictions)}
    2. Başlangıç Tarihi: {dates[0].strftime('%Y-%m-%d')}
    3. Bitiş Tarihi: {dates[-1].strftime('%Y-%m-%d')}
    4. Mevcut Fiyat: ${last_price:.2f}
    5. Tahmin Edilen Son Fiyat: ${predictions[-1]:.2f}
    6. Toplam Beklenen Değişim: {((predictions[-1] - last_price) / last_price * 100):.2f}%

    Günlük Tahmin Detayları:
    {'='*50}"""

                # Her gün için tahmin detaylarını ekle
                for i, (date, pred) in enumerate(zip(dates, predictions)):
                    daily_change = ((pred - (predictions[i-1] if i > 0 else last_price)) / 
                                (predictions[i-1] if i > 0 else last_price) * 100)
                    total_change = ((pred - last_price) / last_price * 100)
                    
                    future_analysis_text += f"""
    Tarih: {date.strftime('%Y-%m-%d')}
    Tahmini Fiyat: ${pred:.2f}
    Günlük Değişim: %{daily_change:.2f}
    Toplam Değişim: %{total_change:.2f}
    {'='*50}"""

                future_analysis_text += """
    \nTahmin Yöntemi:
    1. Mevcut fiyat verilerinden yola çıkıldı
    2. Her gün için bir önceki günün tahminleri kullanıldı
    3. Fiyat değişimleri için %±2 volatilite varsayıldı
    4. İşlem hacmi sabit tutuldu

    Tahmin Modeli Hakkında:
    - Model tipi: Lineer Regresyon
    - Kullanılan özellikler: Açılış Fiyatı, En Yüksek, En Düşük, İşlem Hacmi
    - Eğitim verisi: Son 6 aylık Bitcoin fiyat verileri

    Not: Bu tahminler sadece matematiksel bir projeksiyondur.
    Kripto para piyasaları çok volatil olduğundan, gerçek
    fiyatlar bu tahminlerden önemli ölçüde sapabilir."""

                # Grafiği güncelle
                self.ax.clear()
                
                # Gerçek verileri çiz
                real_data = self.df['Close']
                self.ax.plot(real_data.index, real_data.values, 
                            label='Gerçek Veriler', linewidth=2)
                
                # Tahmin verilerini çiz
                self.ax.plot(future_df['Tarih'], future_df['Tahmin'], 
                            '--', label='Gelecek Tahminleri', linewidth=2)
                
                self.ax.legend()
                self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
                self.ax.grid(True)
                self.canvas.draw()

                # Analiz penceresini göster
                dialog = AnalysisDialog(future_analysis_text, self)
                dialog.exec_()

                return future_df

            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Gelecek tahmini hatası: {str(e)}")
                return pd.DataFrame()

    def make_prediction(self):
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
                                                           label='Gelecek Tahminleri',
                                                           style='--',
                                                           linewidth=2)
            self.ax.legend()
            self.ax.set_title('Bitcoin (BTC) Fiyat Grafiği ve Tahminler')
            self.ax.grid(True)
            self.canvas.draw()
            
            # Sonuçları metin alanına yaz
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BitcoinPredictorApp()
    window.show()
    sys.exit(app.exec_())