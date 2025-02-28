import investpy

df = investpy.get_currency_cross_historical_data(currency_cross="EUR/USD",
                                                 from_date="01/01/2023",
                                                 to_date="01/02/2025",
                                                 interval="Daily")
df.to_csv("eur_usd_historico.csv")
print(df.head())
