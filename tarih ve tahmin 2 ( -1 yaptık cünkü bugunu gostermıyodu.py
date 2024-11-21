import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

def download_data(op, start_date, end_date):
    df = yf.download(op, start=start_date, end=end_date, progress=False)
    return df

def model_engine(model, num):
    df = data[['Close']].copy()  # DataFrame'in bir kopyasını aldık
    
    # `preds` sütununu ekliyoruz, .loc kullanarak uyarıyı önlüyoruz
    df.loc[:, 'preds'] = df['Close'].shift(-num)
    
    print(df)

stock = "ASELS.IS"
today = datetime.date.today()
duration = 100  # bu kadar gün önceki veriyi çekiyoruz
before = today - datetime.timedelta(days=duration)

start_date = before
end_date = today

data = download_data(stock, start_date, end_date)

num = 1
scaler = StandardScaler()
engine = LinearRegression()

model_engine(engine, num)

#çıkan son değerde bi gün sonrası olmadığı için kaydıramıyor
