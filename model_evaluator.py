import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from datetime import datetime, timedelta

class ModelEvaluator:
    def __init__(self, model_instance):
        self.model = model_instance
        self.evaluation_metrics = {}
        self.predictions_df = None

    def evaluate_specific_prediction(self, prediction_date, target_date):
        """
        Belirli bir tahmin için performans değerlendirmesi
        """
        try:
            # Tarihleri datetime'a çevir
            if isinstance(prediction_date, str):
                prediction_date = pd.to_datetime(prediction_date)
            if isinstance(target_date, str):
                target_date = pd.to_datetime(target_date)
            
            # Tahmin tarihine kadar olan verileri al
            historical_data = self.model.df[self.model.df.index <= prediction_date].copy()
            
            # Gerçek değeri al
            actual_price = float(self.model.df.loc[self.model.df.index == target_date, 'Close'].iloc[0])
            
            # Tahmin yap
            self.model.df = historical_data
            predictions = self.model.predict_next_days(target_date.strftime('%Y-%m-%d'))
            target_prediction = float(predictions[predictions['Tarih'] == target_date]['Tahmin'].iloc[0])
            
            # Modeli eski haline getir
            self.model.prepare_data_and_models()
            
            # Hata hesapla
            error = abs(actual_price - target_prediction)
            error_percentage = (error / actual_price) * 100
            
            # Sonuçları hazırla
            results = {
                'prediction_date': prediction_date,
                'target_date': target_date,
                'actual_price': actual_price,
                'predicted_price': target_prediction,
                'error': error,
                'error_percentage': error_percentage,
                'days_difference': (target_date - prediction_date).days
            }
            
            return results
            
        except Exception as e:
            print(f"Değerlendirme hatası: {str(e)}")
            raise e        
        
    def calculate_metrics(self, test_data, predictions, confidence_threshold=0.7):
        """
        Temel tahmin metriklerini hesaplar
        """
        actual_prices = test_data['Close'].values
        pred_prices = predictions['Tahmin'].values
        
        # Temel metrikler
        self.evaluation_metrics['RMSE'] = np.sqrt(mean_squared_error(actual_prices, pred_prices))
        self.evaluation_metrics['MAE'] = mean_absolute_error(actual_prices, pred_prices)
        self.evaluation_metrics['R2'] = r2_score(actual_prices, pred_prices)
        
        # Yüzdesel hata
        mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
        self.evaluation_metrics['MAPE'] = mape
        
        # Yön tahmini başarısı
        actual_direction = np.sign(np.diff(actual_prices))
        pred_direction = np.sign(np.diff(pred_prices))
        direction_accuracy = np.mean(actual_direction == pred_direction) * 100
        self.evaluation_metrics['Direction_Accuracy'] = direction_accuracy
        
        # Yüksek güvenli tahminlerin performansı
        if 'Güven' in predictions.columns:
            high_conf_mask = predictions['Güven'] >= confidence_threshold
            if np.any(high_conf_mask[:-1]):  # En az bir yüksek güvenli tahmin varsa
                high_conf_actual = actual_prices[high_conf_mask]
                high_conf_pred = pred_prices[high_conf_mask]
                self.evaluation_metrics['High_Confidence_MAPE'] = np.mean(
                    np.abs((high_conf_actual - high_conf_pred) / high_conf_actual)
                ) * 100
        
        return self.evaluation_metrics
    
    def calculate_profit_metrics(self, test_data, predictions, initial_investment=10000):
        """
        Karlılık metriklerini hesaplar
        """
        df = pd.DataFrame({
            'Actual': test_data['Close'].values,
            'Predicted': predictions['Tahmin'].values,
            'Date': predictions['Tarih'].values
        })
        
        # Alım-satım sinyalleri
        df['Actual_Return'] = df['Actual'].pct_change()
        df['Predicted_Return'] = df['Predicted'].pct_change()
        
        # Basit alım-satım stratejisi
        df['Position'] = np.where(df['Predicted_Return'] > 0, 1, -1)
        
        # Karlılık hesaplaması
        df['Strategy_Return'] = df['Position'].shift(1) * df['Actual_Return']
        df['Strategy_Return'] = df['Strategy_Return'].fillna(0)
        
        # Kümülatif getiriler
        df['Cumulative_Market_Return'] = (1 + df['Actual_Return']).cumprod()
        df['Cumulative_Strategy_Return'] = (1 + df['Strategy_Return']).cumprod()
        
        # Portföy değerleri
        df['Market_Portfolio'] = initial_investment * df['Cumulative_Market_Return']
        df['Strategy_Portfolio'] = initial_investment * df['Cumulative_Strategy_Return']
        
        # Karlılık metrikleri
        profit_metrics = {
            'Total_Return': (df['Strategy_Portfolio'].iloc[-1] / initial_investment - 1) * 100,
            'Market_Return': (df['Market_Portfolio'].iloc[-1] / initial_investment - 1) * 100,
            'Sharpe_Ratio': self._calculate_sharpe_ratio(df['Strategy_Return']),
            'Max_Drawdown': self._calculate_max_drawdown(df['Strategy_Portfolio']),
            'Win_Rate': np.mean(df['Strategy_Return'] > 0) * 100
        }
        
        self.evaluation_metrics.update(profit_metrics)
        self.predictions_df = df
        
        return profit_metrics
    
    def _calculate_sharpe_ratio(self, returns, risk_free_rate=0.02):
        """Sharpe oranı hesaplama"""
        excess_returns = returns - risk_free_rate/252  # Günlük risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / excess_returns.std())
    
    def _calculate_max_drawdown(self, portfolio_values):
        """Maksimum drawdown hesaplama"""
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdowns = (portfolio_values - rolling_max) / rolling_max
        return np.min(drawdowns) * 100
    
    def generate_evaluation_report(self):
        """Değerlendirme raporu oluşturur"""
        if not self.evaluation_metrics:
            return "Henüz değerlendirme yapılmamış."
        
        report = "Bitcoin Tahmin Modeli Performans Raporu\n"
        report += "=" * 50 + "\n\n"
        
        # Tahmin doğruluğu metrikleri
        report += "1. Tahmin Doğruluğu Metrikleri:\n"
        report += f"   - RMSE: ${self.evaluation_metrics['RMSE']:,.2f}\n"
        report += f"   - MAE: ${self.evaluation_metrics['MAE']:,.2f}\n"
        report += f"   - MAPE: %{self.evaluation_metrics['MAPE']:.2f}\n"
        report += f"   - R2 Skoru: {self.evaluation_metrics['R2']:.3f}\n"
        report += f"   - Yön Tahmini Başarısı: %{self.evaluation_metrics['Direction_Accuracy']:.2f}\n"
        
        if 'High_Confidence_MAPE' in self.evaluation_metrics:
            report += f"   - Yüksek Güvenli Tahminlerde MAPE: %{self.evaluation_metrics['High_Confidence_MAPE']:.2f}\n"
        
        # Karlılık metrikleri
        if 'Total_Return' in self.evaluation_metrics:
            report += "\n2. Karlılık Metrikleri:\n"
            report += f"   - Toplam Getiri: %{self.evaluation_metrics['Total_Return']:.2f}\n"
            report += f"   - Piyasa Getirisi: %{self.evaluation_metrics['Market_Return']:.2f}\n"
            report += f"   - Sharpe Oranı: {self.evaluation_metrics['Sharpe_Ratio']:.3f}\n"
            report += f"   - Maksimum Drawdown: %{self.evaluation_metrics['Max_Drawdown']:.2f}\n"
            report += f"   - Kazanç Oranı: %{self.evaluation_metrics['Win_Rate']:.2f}\n"
        
        return report

def evaluate_model(app_instance):
    """
    Bitcoin tahmin uygulamasının performansını değerlendirir
    """
    # Test verisi oluştur (son 30 gün)
    test_start_date = app_instance.df.index[-30]
    test_data = app_instance.df[app_instance.df.index >= test_start_date]
    
    # Test tarihleri için tahminler al
    predictions = app_instance.predict_next_days(test_data.index[-1].strftime('%Y-%m-%d'))
    
    # Değerlendirici oluştur
    evaluator = ModelEvaluator(app_instance)
    
    # Metrikleri hesapla
    evaluator.calculate_metrics(test_data, predictions)
    evaluator.calculate_profit_metrics(test_data, predictions)
    
    # Rapor oluştur ve göster
    report = evaluator.generate_evaluation_report()
    
    # Sonuçları göster
    dialog = app_instance.__class__.AnalysisDialog(report, app_instance)
    dialog.exec_()
    
    return evaluator