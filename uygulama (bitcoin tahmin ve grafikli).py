import pandas as pd  
import yfinance as yf  
import numpy as np  
from datetime import datetime, timedelta  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LinearRegression  
import matplotlib.pyplot as plt  
import tkinter as tk  
from tkinter import messagebox  

def fetch_data(ticker, start_date, end_date):  
    """Gerekli verileri Yahoo Finance'den çekmek için fonksiyon."""  
    data = yf.download(ticker, start=start_date, end=end_date)  
    return data  

def predict_next_days(end_date, df, model, scaler):  
    """Gelecek günlerin tahminlerini yapar."""  
    last_real_date = df.index[-1].tz_localize(None)  
    target_date = datetime.strptime(end_date, '%Y-%m-%d')  
    days_to_predict = (target_date - last_real_date).days  
    
    if days_to_predict <= 0:  
        return pd.DataFrame()  # Geçerli bir tarih değilse boş DataFrame döndür  
    
    predictions = []  
    dates = []  
    current_features = df[['Open', 'High', 'Low', 'Volume']].iloc[-1:].values  
    
    for i in range(days_to_predict):  
        current_scaled = scaler.transform(current_features)  
        next_price = model.predict(current_scaled)[0]  
        
        next_date = last_real_date + timedelta(days=i+1)  
        predictions.append(next_price)  
        dates.append(next_date.strftime('%Y-%m-%d'))  
        
        current_features = np.array([  
            [float(next_price),  # Open  
             float(next_price * 1.01),  # High  
             float(next_price * 0.99),  # Low  
             float(current_features[0][3])]  # Volume  
        ])  
    
    return pd.DataFrame({  
        'Tarih': dates,  
        'Tahmin Edilen Fiyat': predictions  
    })  

def run_prediction():  
    """Butona basıldığında çalışacak fonksiyon."""  
    end_date = entry_date.get()  
    today = datetime.now().strftime('%Y-%m-%d')  
    
    if end_date < today:  
        messagebox.showwarning("Uyarı", "Tahmin tarihi bugünden geçmiş olamaz.")  
        return  

    predictions_df = predict_next_days(end_date, df, model, scaler)  
    
    if predictions_df.empty:  
        messagebox.showwarning("Uyarı", "Tahmin yapılamadı, lütfen tarihi kontrol edin.")  
        return  
    
    # Tahmin sonuçlarını görüntüle  
    result_text = "\nTahminler:\n"  
    for _, row in predictions_df.iterrows():  
        result_text += f"Tarih: {row['Tarih']}, Tahmin: ${row['Tahmin Edilen Fiyat']:.2f}\n"  
    
    messagebox.showinfo("Tahmin Sonuçları", result_text)  

def plot_data():  
    """Bitcoin fiyatlarını çizen fonksiyon."""  
    plt.figure(figsize=(10, 5))  
    plt.plot(df['Close'], label='Kapanış Fiyatı', color='blue')  
    plt.title('Bitcoin (BTC) Kapanış Fiyatı Grafiği')  
    plt.xlabel('Tarih')  
    plt.ylabel('Fiyat (USD)')  
    plt.legend()  
    plt.show()  

# Uygulama başlangıç ayarları  
ticker = "BTC-USD"  
start_date = "2024-04-01"  
end_date = datetime.now().strftime("%Y-%m-%d")  

# Veri setini yükle  
df = fetch_data(ticker, start_date, end_date)  

# Hedef değişkeni ve model eğitimi  
num = 1  
df.loc[:, 'preds'] = df['Close'].shift(-num)  
df.dropna(inplace=True)  

X = df[['Open', 'High', 'Low', 'Volume']]  
y = df['preds']  

scaler = StandardScaler()  
X_scaled = scaler.fit_transform(X)  

model = LinearRegression()  
model.fit(X_scaled, y)  

# Tkinter arayüzü ayarları  
root = tk.Tk()  
root.title("Bitcoin Tahmin Uygulaması")  

# Tarih Girişi için Etiket ve Giriş Kutusu  
label_date = tk.Label(root, text="Tahmin Yapılacak Tarih (YYYY-MM-DD):")  
label_date.pack()  

entry_date = tk.Entry(root)  # Kullanıcının tarih gireceği alan  
entry_date.pack()  

# Tahmin butonu  
btn_predict = tk.Button(root, text="Tahmin Et", command=run_prediction)  
btn_predict.pack()  

# Grafiği göster butonu  
btn_plot = tk.Button(root, text="Bitcoin Grafiğini Göster", command=plot_data)  
btn_plot.pack()  

# Uygulama döngüsü  
root.mainloop()