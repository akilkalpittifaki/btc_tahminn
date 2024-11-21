from binance.client import Client
import pandas as pd
import time

# Binance API Anahtarlarınızı Ekleyin
API_KEY = 'u1AVO8mPWW3pMoqth8YhcbOV5gqfz3w6s2FFmbtB3RL5yY63g4PTKsTsVdzvV6P9'
SECRET_KEY = 'xp9gxcooNLsKPgmvVrWIikVM1LkeiEA1QujK6Rd7sjgveC2Oga2Wspnd0Tkc5JKu'  # Secret Key'inizi buraya yazın

client = Client(API_KEY, SECRET_KEY)

# RSI Hesaplama Fonksiyonu
def calculate_rsi(data, period=14):
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Geçmiş Veriyi Çekme
def get_historical_data(symbol, interval, lookback):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=lookback)
    data = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    data['close'] = data['close'].astype(float)
    return data

# Bitcoin Al/Sat İşlemi
def trade(symbol, quantity, rsi):
    if rsi.iloc[-1] < 30:  # RSI 30'un altındaysa AL
        print(f"{symbol} alınacak! RSI: {rsi.iloc[-1]}")
        # client.order_market_buy(symbol=symbol, quantity=quantity)
    elif rsi.iloc[-1] > 70:  # RSI 70'in üstündeyse SAT
        print(f"{symbol} satılacak! RSI: {rsi.iloc[-1]}")
        # client.order_market_sell(symbol=symbol, quantity=quantity)

# Ana Döngü
def main():
    symbol = 'BTCUSDT'  # İşlem çifti
    interval = Client.KLINE_INTERVAL_15MINUTE  # 15 dakikalık mum verisi
    lookback = 100  # Geriye dönük veri miktarı
    quantity = 0.001  # İşlem miktarı (örneğin 0.001 BTC)

    while True:
        data = get_historical_data(symbol, interval, lookback)
        data['rsi'] = calculate_rsi(data)
        trade(symbol, quantity, data['rsi'])
        time.sleep(60 * 15)  # Her 15 dakikada bir çalışır

if __name__ == "__main__":
    main()
