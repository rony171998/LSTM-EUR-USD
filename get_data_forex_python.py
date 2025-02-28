from forex_python.converter import CurrencyRates
import datetime
import pandas as pd

c = CurrencyRates()
start_date = datetime.datetime(2023, 1, 1)
end_date = datetime.datetime(2025, 2, 1)
delta = datetime.timedelta(days=1)

data = []
while start_date <= end_date:
    try:
        rate = c.get_rate('EUR', 'USD', start_date)
        data.append([start_date.strftime("%Y-%m-%d"), rate])
    except:
        pass
    start_date += delta

df = pd.DataFrame(data, columns=["Date", "Rate"])
df.to_csv("eur_usd_historico.csv", index=False)
print(df.head())
