import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import plotly.graph_objects as go

bitcoin = pd.read_csv("BTCUSD_daily.csv", index_col = 0, parse_dates=['date'])
print(bitcoin.head())
#курс биткоина
fig = go.Figure(data=[go.Candlestick(x=bitcoin['date'],
                open=bitcoin['open'],
                high=bitcoin['high'],
                low=bitcoin['low'],
                close=bitcoin['close'])])
#fig.show()
#разница high и low
plt.plot(bitcoin.date, bitcoin.high - bitcoin.low)
plt.title('high-low')
plt.xlabel('date')
plt.ylabel('USD cost')
plt.grid()
plt.show()
#объем продаж
plt.ticklabel_format(useOffset=False, style='plain')
plt.plot(bitcoin['date'], bitcoin['Volume USD'])
plt.title('Volume')
plt.xlabel('date')
plt.ylabel('USD cost')
plt.grid()
plt.show()
for day in range(1,8):
    # создаем колонку: какая была цена close по состоянию на {day} дней назад
    bitcoin[f"close_{day}d"] = bitcoin["close"].shift(day)
# Чистим данные
print(bitcoin.head())
date = bitcoin[:-1].date
#print(date) #2651 строк - 1 столбец
bitcoin.drop("symbol", axis=1, inplace=True)
bitcoin.drop("date", axis=1, inplace=True)
bitcoin.fillna(method="backfill", inplace=True)
# target = цена закрытия на завтра
bitcoin["target"] = bitcoin.close.shift(-1)
X = bitcoin[:-1].drop("target", axis=1)
y = bitcoin[:-1].target
#print(y) #2651 строк - 1 столбец
#sklearn учим и тестируем модель
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
print("Train set:")
print(X_train.shape)
print(y_train.shape)
print("Test set:")
print(X_test.shape)
print(y_test.shape)
# Создаем линейную модель
model = LinearRegression()
model.fit(X_train, y_train) # Обучаем
y_pred = model.predict(X_test) # Просим модель предсказать для X_test -> y_pred
#print('На завтрашний день курс биткоина будет равен', y_pred)
# Средняя абсолютная ошибка:
mae = (y_pred - y_test).abs().mean()
# Средняя ошибка:
me = (y_pred - y_test).mean()
print(f"В среднем модель ошиблась на {mae:.0f} долларов в день")
print(f"В среднем модель ошиблась на {me:.0f} долларов")
#график предсказывания лин модели
rows = y_test.index
y_date = date[rows]
#print(y_date)
plt.plot(y_date, y_pred, 'bo', alpha=0.3)
plt.plot(date, bitcoin.close[:-1], 'red', alpha=0.7)
plt.title('Truth or Bluff')
plt.xlabel('date')
plt.ylabel('USD cost')
plt.grid()
plt.show()