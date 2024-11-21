import pandas as pd
import matplotlib.pyplot as plt
import re
from datetime import datetime
import matplotlib.dates as mdates

# Verileri al ve işle
def parse_prediction_data(data):
    tarih_list = []
    tahmin_list = []

    pattern = r"Tarih: (\d{4}-\d{2}-\d{2})\\nTahmin: \$(\d+[\d,]*\.\d{2})"
    matches = re.findall(pattern, data)

    for match in matches:
        tarih_str, tahmin_str = match
        tarih = datetime.strptime(tarih_str, "%Y-%m-%d")
        tahmin = float(tahmin_str.replace(',', ''))

        tarih_list.append(tarih)
        tahmin_list.append(tahmin)

    return tarih_list, tahmin_list

# Grafik oluşturma fonksiyonu
def create_bitcoin_prediction_chart(tarih_list, tahmin_list):
    plt.figure(figsize=(12, 6))
    plt.plot(tarih_list, tahmin_list, marker='o', linestyle='-', color='b')

    plt.xlabel("Tarih")
    plt.ylabel("Tahmin ($)")
    plt.title("Bitcoin Fiyat Tahmin Grafiği")
    plt.xticks(rotation=45)
    plt.grid(True)
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    plt.tight_layout()
    plt.show()

# Tahmin verisi
data = """Tarih: 2024-11-10\nTahmin: $65,262.04\nTrend: accumulation\nGüven: %94.1\n\nTarih: 2024-11-11\nTahmin: $55,448.14\nTrend: accumulation\nGüven: %96.0\n\n..."""

# Veriyi işle
tarih_list, tahmin_list = parse_prediction_data(data)

# Grafik oluştur
tarihi_plot = create_bitcoin_prediction_chart(tarih_list, tahmin_list)
