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
            start_date = (target_date - timedelta(days=180)).strftime('%Y-%m-%d')
            end_date = target_date.strftime('%Y-%m-%d')
            
            try:
                df_past = yf.download(self.ticker, start=start_date, end=end_date)
                df_past.loc[:, 'preds'] = df_past['Close'].shift(-1)
                df_past.dropna(inplace=True)
                
                X_past = df_past[['Open', 'High', 'Low', 'Volume']]
                actual_prices = df_past['Close']
                
                X_past_scaled = self.scaler.transform(X_past)
                predicted_prices = self.model.predict(X_past_scaled)
                
                accuracy, analysis_text = self.calculate_accuracy_metrics(actual_prices, predicted_prices)
                
                # Grafiği güncelle
                self.ax.clear()
                self.ax.plot(df_past.index, actual_prices, label='Gerçek Veriler', linewidth=2)
                self.ax.plot(df_past.index, predicted_prices, '--', label=f'Model Tahminleri (Doğruluk: {accuracy:.2f}%)', linewidth=2)
                self.ax.set_title('Bitcoin (BTC) Geçmiş Fiyat Analizi')
                self.ax.legend()
                self.ax.grid(True)
                self.canvas.draw()
                
                # Analiz penceresini göster
                dialog = AnalysisDialog(analysis_text, self)
                dialog.exec_()
                
                return pd.DataFrame()
                
            except Exception as e:
                QMessageBox.critical(self, "Hata", f"Geçmiş veri tahmini hatası: {str(e)}")
                return pd.DataFrame()

        # Gelecek tahminleri
        predictions = []
        dates = []
        current_features = self.df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values

        for i in range(days_to_predict):
            current_scaled = self.scaler.transform(current_features)
            next_price = self.model.predict(current_scaled)[0]
            
            next_date = last_real_date + timedelta(days=i+1)
            predictions.append(next_price)
            dates.append(next_date)
            
            current_features = np.array([[
                float(next_price),
                float(next_price * 1.01),
                float(next_price * 0.99),
                float(current_features[0][3])
            ]])

        future_df = pd.DataFrame({
            'Tarih': dates,
            'Tahmin': predictions
        })

        # Tahmin analizi metnini oluştur
        analysis_text = f"""Gelecek Tahmin Analizi:

1. Tahmin Edilen Gün Sayısı: {len(predictions)}
2. Başlangıç Tarihi: {dates[0].strftime('%Y-%m-%d')}
3. Bitiş Tarihi: {dates[-1].strftime('%Y-%m-%d')}
4. Başlangıç Fiyatı: ${predictions[0]:.2f}
5. Tahmin Edilen Son Fiyat: ${predictions[-1]:.2f}
6. Toplam Değişim: {((predictions[-1] - predictions[0]) / predictions[0] * 100):.2f}%

Tahmin Modeli Hakkında:
- Kullanılan model: Lineer Regresyon
- Eğitim verisi: {self.df.index[0].strftime('%Y-%m-%d')} - {self.df.index[-1].strftime('%Y-%m-%d')}
- Kullanılan özellikler: Açılış, En Yüksek, En Düşük, İşlem Hacmi

Not: Bu tahminler geçmiş verilere dayalı istatistiksel bir projeksiyon sunar.
Kripto para piyasaları yüksek volatiliteye sahiptir ve beklenmedik 
olaylar fiyatları önemli ölçüde etkileyebilir."""

        # Analiz penceresini göster
        dialog = AnalysisDialog(analysis_text, self)
        dialog.exec_()

        return future_df

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