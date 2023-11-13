import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

# Wczytaj dane historyczne
df_hist = pd.read_csv("C:\\Pliki_MGR\\pogoda_histUltra.csv")
df_hist['Date'] = pd.to_datetime(df_hist['Date'], format="%m-%d-%Y")

# Wczytaj dane testowe
df_test = pd.read_csv("C:\\Pliki_MGR\\pogoda_test.csv")
df_test['Date'] = pd.to_datetime(df_test['Date'], format="%m-%d-%Y")

# Inicjalizacja wyników
results = []

# Inicjalizacja list do przechowywania prawdziwych i przewidywanych wartości dla każdego dnia
all_true = [[] for _ in range(7)]
all_pred = [[] for _ in range(7)]

# Inicjalizacja skalera
scaler = MinMaxScaler(feature_range=(0,1))

# Przetwarzanie dla każdego miasta
for city in df_hist['City'].unique():
    df_city = df_hist[df_hist['City'] == city].sort_values('Date')
    
    # Normalizacja danych
    data_scaled = scaler.fit_transform(df_city['Temperature'].values.reshape(-1, 1))
    
    # Tworzenie sekwencji
    print(f"Tworzenie sekwencji dla {city}...")
    X = []
    Y = []
    for i in range(7, len(data_scaled)): # rozmiar sekwencji, liczba dni historycznych które model bierze pod uwagę  //default 7
        X.append(data_scaled[i-7:i])
        Y.append(data_scaled[i])
    X = np.asarray(X).astype('float32')
    Y = np.asarray(Y).astype('float32')
    
    # Tworzenie modelu LSTM
    print(f"Tworzenie modelu LSTM dla {city}...")
    model = Sequential()
    model.add(LSTM(100, activation='relu', input_shape=(7, 1))) # liczba neuronów w warstwie LSTM   // deafult 50, ale do testowania
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.005), loss='mse') # częstość uczenia (zmniejszyć jeżeli model nie zbiega, zwiększyć jeżeli uczenie jest zbyt wolne) // default 0.01
    
    # Trenowanie modelu
    print(f"Trenowanie modelu LSTM dla {city}...")
    model.fit(X.reshape(X.shape[0], X.shape[1], 1), Y, epochs=100, verbose=0) # liczba epok  //default 100
    
    try:
        # Przewidywanie temperatury na kolejne 7 dni
        print(f"Przewidywanie temperatury na kolejne 7 dni dla {city}...")
        x_test = scaler.transform(df_test[df_test['City'] == city].iloc[0, 2:].values.astype('float32').reshape(-1, 1))
        pred_values = []
        for _ in range(7):
            y_pred = model.predict(x_test.reshape(1, 7, 1))
            pred_values.append(np.squeeze(y_pred))
            x_test = np.roll(x_test, -1)
            x_test[-1] = y_pred

        results.append([city] + list(scaler.inverse_transform(np.array(pred_values).reshape(-1, 1)).flatten()))
        
        # Dodawanie prawdziwych i przewidywanych wartości do odpowiednich list
        for i in range(7):
            all_true[i].append(df_city['Temperature'].values[-7+i])
            all_pred[i].append(scaler.inverse_transform(np.array(pred_values[i]).reshape(-1, 1)).flatten()[0])
            
    except Exception as e:
        print(f"Wystąpił błąd dla miasta {city}: {str(e)}")
        results.append([city] + [0]*7)

# Zapisywanie wyników do pliku
df_results = pd.DataFrame(results, columns=["City","Day1","Day2","Day3","Day4","Day5","Day6","Day7"])
df_results.to_csv("C:\\Pliki_MGR\\predykcja_LSTM.csv", index=False)

# Obliczanie i wyświetlanie dokładności dla każdego dnia
for i in range(7):
    accuracy = mean_absolute_error(all_true[i], all_pred[i])
    print(f"Dokładność dla dnia {i+1}: {accuracy}")
