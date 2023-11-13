import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Wczytaj dane
historical_data = pd.read_csv('C:\\Pliki_MGR\\pogoda_histUltra.csv')
actual_data = pd.read_csv('C:\\Pliki_MGR\\pogoda_test.csv')
historical_data['Date'] = pd.to_datetime(historical_data['Date'], format='%m-%d-%Y')
actual_data['Date'] = pd.to_datetime(actual_data['Date'], format='%m-%d-%Y')

# Zignoruj ostrzeżenia
warnings.filterwarnings('ignore')

# Słownik do przechowywania prognoz
forecasts = {}

# Kolejność miast
cities_order = ['Lublin', 'Szczecin', 'Olsztyn', 'Rzeszów', 'Kraków', 
                'Gorzów Wielkopolski', 'Wrocław', 'Białystok', 'Gdańsk', 
                'Opole', 'Katowice', 'Warszawa', 'Toruń', 'Poznań', 
                'Kielce', 'Łódź']

for city in cities_order:
    city_data = historical_data[historical_data['City'] == city].sort_values('Date')
    city_data.set_index('Date', inplace=True)

    # parametry p,d,q
    p = 10
    d = 3
    q = 0

    # Model ARIMA
    model = ARIMA(city_data['Temperature'], order=(p, d, q))

    # Dopasowanie modelu
    model_fit = model.fit()

    # Prognozowanie dla 7 dni
    forecast_results = model_fit.forecast(steps=7)
    forecast = forecast_results.values

    forecasts[city] = forecast

# Zamień prognozy w ramkę danych
forecasts_df = pd.DataFrame(forecasts).transpose()
forecasts_df.columns = ['Day1', 'Day2', 'Day3', 'Day4', 'Day5', 'Day6', 'Day7']
forecasts_df.index.name = 'City'

# Obliczanie błędu średniokwadratowego dla każdego dnia
mses = []
for i in range(1, 8):
    # Filtruj dane rzeczywiste tylko dla miast, które istnieją w forecasts_df
    actual_data_filtered = actual_data[actual_data['City'].isin(forecasts_df.index)]
    mse = mean_squared_error(actual_data_filtered[f'Temperature_Day{i}'], forecasts_df.loc[actual_data_filtered['City']].iloc[:, i-1])
    mses.append(mse)

# Obliczanie średniej z błędów
accuracy = 100 - np.mean(mses)

# Zapisanie dokładności do pliku
with open('C:\\Pliki_MGR\\accuracy.csv', 'w') as f:
    f.write(f"Accuracy,{accuracy}")

# Tworzenie wykresu dla Wrocław
plt.plot(range(1, 8), forecasts['Wrocław'])
plt.title('Prognoza dla Wrocław')
plt.xlabel('Dzień')
plt.ylabel('Temperatura')
plt.show()

# Zapisanie prognoz do pliku
forecasts_df.to_csv('C:\\Pliki_MGR\\predykcja.csv', float_format='%.2f')
print("Prognozy zapisano do 'predykcjacsv'")
