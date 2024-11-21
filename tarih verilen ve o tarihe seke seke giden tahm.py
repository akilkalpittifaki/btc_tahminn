# Gerekli kütüphanelerin içe aktarılması  
# pandas: Veri manipülasyonu için  
# yfinance: Yahoo Finance'den finansal veri çekmek için  
# StandardScaler: Verileri normalize etmek için  
# LinearRegression: Doğrusal regresyon modeli için  
# numpy: Sayısal işlemler için  
# datetime: Tarih işlemleri için  
import pandas as pd  
import yfinance as yf  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LinearRegression  
import numpy as np  
from datetime import datetime, timedelta  

# Gelecek günlerin fiyatlarını tahmin eden fonksiyon  
def predict_next_days(end_date, df, model, scaler):  
    """  
    Bu fonksiyon, eğitilmiş modeli kullanarak gelecek günlerin Bitcoin fiyatlarını tahmin eder.  
    Tahmin mekanizması şu şekilde çalışır:  
    1. Her gün için bir önceki günün tahminini kullanarak yeni tahmin yapar  
    2. Yeni tahminler için özellikler (Open, High, Low, Volume) güncellenir  
    3. Her tahmin, bir önceki tahminin sonucuna dayanır  
    """  
    # Son gerçek veri tarihini al ve timezone'u kaldır  
    last_real_date = df.index[-1].tz_localize(None)  
    
    # Hedef tarihe kadar kaç gün tahmin yapılacağını hesapla  
    target_date = datetime.strptime(end_date, '%Y-%m-%d')  
    days_to_predict = (target_date - last_real_date).days  
    
    if days_to_predict <= 0:  
        return pd.DataFrame()  # Hedef tarih geçmiş tarihse boş DataFrame dön  
    
    # Tahminleri ve tarihleri saklamak için boş listeler  
    predictions = []  
    dates = []  
    # İlk tahmin için son gerçek verileri al  
    current_features = df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values  
    
    # Her gün için tahmin döngüsü  
    for i in range(days_to_predict):  
        # Verileri ölçekle ve tahmin yap  
        # StandardScaler ile normalize edilmiş veriler modele gönderilir  
        current_scaled = scaler.transform(current_features)  
        # Model tahmini yapar  
        next_price = model.predict(current_scaled)[0]  
        
        # Tahmin sonuçlarını kaydet  
        next_date = last_real_date + timedelta(days=i+1)  
        predictions.append(next_price)  
        dates.append(next_date.strftime('%Y-%m-%d'))  
        
        # Bir sonraki gün için özellikleri güncelle  
        # - Open: Tahmin edilen fiyat  
        # - High: Tahmin edilen fiyat + %1 (volatilite tahmini)  
        # - Low: Tahmin edilen fiyat - %1 (volatilite tahmini)  
        # - Volume: Son bilinen işlem hacmi (sabit tutulur)  
        current_features = np.array([  
            [float(next_price),  # Open price  
             float(next_price * 1.01),  # High price (+1%)  
             float(next_price * 0.99),  # Low price (-1%)  
             float(current_features[0][3])]  # Volume  
        ])  
    
    # Tahminleri DataFrame'e dönüştür ve döndür  
    return pd.DataFrame({  
        'Tarih': dates,  
        'Tahmin Edilen Fiyat': predictions  
    })  

# Bitcoin verilerini Yahoo Finance'den çek  
# BTC-USD çifti seçildi ve belirli tarih aralığı için veriler indirildi  
ticker = "BTC-USD"  
data = yf.download(ticker, start="2024-04-01", end="2024-11-09")  
df = pd.DataFrame(data)  

# Hedef değişkeni oluştur  
# 'preds' sütunu, bir sonraki günün kapanış fiyatını temsil eder  
num = 1  # Bir gün sonrası için tahmin  
df.loc[:, 'preds'] = df['Close'].shift(-num)  # Kapanış fiyatını bir gün kaydır  
df.dropna(inplace=True)  # Eksik verileri temizle  

# Özellikler (X) ve hedef (y) değişkenlerini ayır  
# X: Modelin öğreneceği özellikler (Open, High, Low, Volume)  
# y: Modelin tahmin edeceği değer (bir sonraki günün kapanış fiyatı)  
X = df[['Open', 'High', 'Low', 'Volume']]  
y = df['preds']  

# Verileri ölçekle (normalize et)  
# StandardScaler, verileri ortalaması 0 ve varyansı 1 olacak şekilde dönüştürür  
# Bu, modelin daha iyi öğrenmesini sağlar  
scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

# Doğrusal Regresyon modelini eğit  
# Model, X_scaled (özellikler) ile y (hedef) arasındaki ilişkiyi öğrenir  
# LinearRegression, en küçük kareler yöntemini kullanarak en uygun doğruyu bulur  
model = LinearRegression()  
model.fit(X_scaled, y)  # Model eğitimi burada gerçekleşir  

# Belirlenen tarihe kadar tahmin yap  
end_date = "2024-11-09"  
predictions_df = predict_next_days(end_date, df, model, scaler)  

# Model performansını değerlendir  
# R-kare (R²) skoru hesapla: 1'e yakın değerler daha iyi performansı gösterir  
accuracy = model.score(X_scaled, y)  

# Sonuçları ekrana yazdır  
print(f"Son gerçek veri tarihi: {df.index[-1].strftime('%Y-%m-%d')}")  
print(f"\nTahminler:")  
for _, row in predictions_df.iterrows():  
    print(f"Tarih: {row['Tarih']}, Tahmin: ${row['Tahmin Edilen Fiyat']:.2f}")  

print(f"\nModel doğruluğu: {accuracy:.4f}")  

# Tahminleri CSV dosyasına kaydet  
predictions_df.to_csv('btc_future_predictions.csv', index=False)  
print(f"\nTahmin sonuçları 'btc_future_predictions.csv' dosyasına kaydedildi.")  

"""  
Yapay Zeka Tahmin Mekanizması Açıklaması:  

1. Veri Hazırlama:  
   - Bitcoin'in geçmiş verileri (Open, High, Low, Volume) kullanılır  
   - Veriler StandardScaler ile normalize edilir  
   - Her gün için bir sonraki günün kapanış fiyatı hedef olarak belirlenir  

2. Model Eğitimi:  
   - Doğrusal Regresyon modeli, normalize edilmiş veriler üzerinde eğitilir  
   - Model, özellikler ile hedef arasındaki doğrusal ilişkiyi öğrenir  
   - Eğitim sırasında en küçük kareler yöntemi kullanılır  

3. Tahmin Süreci:  
   - Model, son gerçek verilerden başlayarak her gün için tahmin yapar  
   - Her tahmin, bir önceki günün tahmin sonuçlarına dayanır  
   - Yeni tahminler için özellikler güncellenir:  
     * Open: Tahmin edilen fiyat  
     * High: Tahmin + %1  
     * Low: Tahmin - %1  
     * Volume: Sabit tutulur  

4. Model Performansı:  
   - R² skoru ile ölçülür (0 ile 1 arası)  
   - 1'e yakın değerler daha iyi performansı gösterir  
"""