import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split  #testi yapar
from sklearn.linear_model import LinearRegression

# Örnek olarak bir veri seti yükleyelim
ticker = "BTC-USD"  # Bitcoin veri kümesini örnek alıyoruz
data = yf.download(ticker, start="2024-09-01", end="2024-10-5")
df = pd.DataFrame(data)
# Bu satırlar, yfinance kütüphanesi aracılığıyla belirli bir tarih aralığındaki Bitcoin fiyat verilerini Yahoo Finance'ten indirir ve bir DataFrame’e (df) aktarır.
# Veri, günlük açılış (Open), kapanış (Close), en yüksek (High), en düşük (Low) ve hacim (Volume) gibi sütunları içerir.

# Özelliklerinizi ve hedef değişkeninizi belirleyin
num = 1  # Tahmin yapılacak gün sayısı

# Hedef değişkeni ekleyin, uyarıyı önlemek için .loc kullanıyoruz
df.loc[:, 'preds'] = df['Close'].shift(-num)

# Bu satır, kapanış fiyatlarını num kadar kaydırarak (örneğin num = 1 için bir gün) hedef değişken (preds) oluşturur. Bu, bir gün sonrasının kapanış fiyatını tahmin etmeyi sağlar.
# Kaydırma (shift) işlemi sonrasında oluşan NaN değerleri kaldırmak için df.dropna(inplace=True) kullanılır.

# NaN değerleri kaldırın (shift işlemi sonucu oluşur)
df.dropna(inplace=True)

# Özellik ve hedef ayrımı
X = df[['Open', 'High', 'Low', 'Volume']]
y = df['preds']

#Özellikler (X): Modelin tahmin yaparken kullanacağı giriş verileridir. Burada açılış, en yüksek, en düşük fiyatlar ve hacim (Volume) özellik olarak seçilmiştir.
#Hedef (y): Modelin tahmin etmeye çalıştığı değerdir, burada preds yani bir gün sonrası kapanış fiyatı.

# Veriyi ölçekleyin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#Bu adımda, özelliklerin (X) daha iyi performans için standartlaştırılması (ölçeklenmesi) sağlanır. 
# StandardScaler, her bir özelliği ortalaması 0, standart sapması 1 olacak şekilde dönüştürür.

# Eğitim ve test setlerini ayırın
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
# Veri seti %80 eğitim ve %20 test olacak şekilde bölünür. 
# Eğitim verisi modelin öğrenmesi için kullanılırken, test verisi modelin doğruluğunu değerlendirmek için ayrı


# Modeli oluşturun ve eğitin
model = LinearRegression()
model.fit(X_train, y_train)
#Linear Regression (Doğrusal Regresyon) modeli oluşturulur ve fit metodu ile eğitim verileri üzerinde model eğitilir. 
# Model, fiyatlar arasında doğrusal bir ilişki olduğunu varsayar.

# Model doğruluğunu kontrol edin
accuracy = model.score(X_test, y_test)
print(f"Model doğruluğu: {accuracy}")
#model.score fonksiyonu, modelin doğruluğunu (burada R^2 skoru) test verileri üzerinde hesaplar.
# örneğinizde 0.9874727196132018 sonucu alınmış, bu da modelin veriyi %98.75 oranında açıkladığı anlamına gelir. 
# Yani, model verilerdeki varyansın %98.75'ini yakalamaktadır; bu oldukça yüksek bir doğruluk oranıdır.

# Tahmin yapın (örneğin, bir sonraki gün kapanış fiyatı tahmini)
predicted_price = model.predict([X_test[0]])  # İlk test verisini kullanarak tahmin yapıyoruz
print(f"Tahmin edilen kapanış fiyatı (1. gün): {predicted_price[0]}")

#predict fonksiyonu ile model, test verilerinden bir örnek (ilk satır) üzerinden bir tahmin yapar.
#Çıktıda Tahmin edilen kapanış fiyatı (1. gün): 26214.847134923923 olarak gösteriliyor. Bu, test veri setindeki ilk gün için tahmin edilen kapanış fiyatıdır.