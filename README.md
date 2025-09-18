import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error

#ДЛЯ БАЗОВОЙ ПОДГОТОВКИ МОДЕЛИ

#ЗАВОДИМ ДАННЫЕ
df = pd.read_excel('fact_drp.xlsx')
colums_list = df.columns.tolist()

df['DATE']  = pd.to_datetime(df['DATE'], format='%d.%m.%Y', errors='coerce')
df['TOTAL'] = pd.to_numeric(df['TOTAL'], errors='coerce')

# СТРОИМ ГРАФИК
df = df.dropna(subset=['DATE', 'TOTAL']).set_index('DATE').asfreq('MS')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['TOTAL'], label='факт.численность', linewidth=3, color = 'green')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

ax.set_title('исходный ряд')
ax.set_xlabel('Месяц'); ax.set_ylabel('ФЧ, ЧЕЛ')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()

#стационарность

def test_stationarity(series, title=''):
    print(f"Results of ADF Test on {title}:")
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("РЯД НЕ СТАЦИОНАРЕН")
    else:
        print("РЯД СТАЦИОНАРЕН")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("\n")

test_stationarity(df['TOTAL'], 'Counts')


# трейн и тест

train_size = int(len(df) * 0.80)

train = df.iloc[:train_size]   # 
test  = df.iloc[train_size:]   # 

print("Размер train:", train.shape)
print("Размер test :", test.shape)

#дифф 1

# первое дифференцирование
train_diff1 = train["TOTAL"].diff().dropna()

# проверка функцией 
print("ADF p-value после 1 diff:", adfuller(train_diff1)[1])
test_stationarity(train_diff1, "Train after 1st diff")

train_diff1.plot()

#дифф 2
# второе дифференцирование
train_diff2 = train["TOTAL"].diff().diff().dropna()

# проверка ADF снова
print("ADF p-value после 2 diff:", adfuller(train_diff2)[1])
test_stationarity(train_diff2, "Train after 2nd diff")
train_diff2.plot()

#дифф 3
# третье дифференцирование
train_diff3 = train["TOTAL"].diff().diff().diff().dropna()

# проверка ADF снова
print("ADF p-value после 3 diff:", adfuller(train_diff3)[1])
test_stationarity(train_diff3, "Train after 2nd diff")
train_diff3.plot()

#ACF
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plot_acf(train_diff3, lags=10)
plt.title("ACF для train_diff3")
plt.show()

#PACF
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt

plt.figure(figsize=(8,4))
plot_pacf(train_diff3, lags=10, method="ywm")  # "ywm" – более устойчивая оценка
plt.title("PACF для train_diff3")
plt.show()

#модель 1
p = 1
d = 3
q = 1

model = ARIMA(train["TOTAL"], order=(p, d, q))
print('--------------------')
model_fit = model.fit()
print('--------------------')
print(model_fit.summary())


#проверка MAPE
forecast = model_fit.forecast(steps=len(test))
mape = np.mean(np.abs((test["TOTAL"] - forecast) / test["TOTAL"])) * 100
print('--------------------')
print(f"MAPE: {mape:.2f}%")

#проверка MAE
y_true = test["TOTAL"]        
y_pred = forecast            
mae = mean_absolute_error(y_true, y_pred)
print('--------------------')
print(f"MAE: {mae:.2f}")

#печать прогнозных значений
forecast = pd.Series(forecast.values, index=test.index, name="forecast")
print('--------------------')
print (forecast)

plt.figure(figsize=(12,6))
plt.plot(train.index, train["TOTAL"], label="Обучающая выборка", linewidth=3, color="green")
plt.plot(test.index, test["TOTAL"], label="Тестовая выборка", linewidth=3, color="y")
plt.plot(test.index, forecast, label="Прогноз", linewidth=3, linestyle='--', color="blue")

plt.title("ПРОВЕРКА МОДЕЛИ -- ПРОГНОЗ VS ТЕСТ")
plt.xlabel("Дата")
plt.ylabel("ФЧ, ЧЕЛ.")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()


#прогноз на 12 мес
p = 1
d = 3
q = 1

model = ARIMA(df["TOTAL"], order=(p, d, q))
model_fit = model.fit()
print('--------------------')
print(model_fit.summary())

s = pd.to_numeric(df["TOTAL"], errors="coerce").dropna() 
s.index = pd.to_datetime(df.index) # если уже datetime — ок 
freq = pd.infer_freq(s.index) 
if freq is None: 
    freq = "MS" # поменяйте при другой периодичности: "M","D","Q" и т.д. 
s = s.asfreq(freq).interpolate() # выровнять и закрыть единичные пропуски

model = ARIMA(s, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False) 
model_fit = model.fit()
print('----------------') 
print(model_fit.summary())

fc = model_fit.get_forecast(steps=12) 
yhat = fc.predicted_mean # значения прогноза ci = fc.conf_int() # доверительные интервалы

#print("\nПрогноз на 12 периодов:") 
#print(yhat)

plt.figure(figsize=(12,6)) 
plt.plot(s.index, s.values, label="Факт") 
plt.plot(yhat.index, yhat.values, label="Прогноз (12)", linewidth=2) 

plt.title("ПРОГНОЗ --БАЗОВАЯ ARIMA") 
plt.xlabel("Дата"); 
plt.ylabel("ФЧ, ЧЕЛ.") 
plt.legend(); 
plt.grid(True, linestyle="--", alpha=0.6) 
plt.show()

#прогноз на 12 + ограничитель
CAP = 350 # <<< ваш порог (константа) 
HALF_YEAR = 6 # полгода (в шагах прогноза)

yhat = fc.predicted_mean.copy() 
ci = fc.conf_int()



reached = (yhat >= CAP).values 
yhat_capped = yhat.copy()

if reached.any(): 
    first_hit_pos = int(np.argmax(reached)) # позиция первого превышения 
    if first_hit_pos < HALF_YEAR: # если достигли порога в первые 6 месяцев — дальше плоско на CAP 
        yhat_capped.iloc[first_hit_pos:] = CAP 
    else: # если порог достигнут позже 6 мес — просто не даём превысить 
        yhat_capped = yhat_capped.clip(upper=CAP) 
else: # порог в горизонте не достигнут — на всякий случай не превышаем порог 
        yhat_capped = yhat_capped.clip(upper=CAP)
        
#---- печать и график с ограничением ----
print("\nПрогноз (c ограничением):") 
print(yhat_capped)

plt.figure(figsize=(12,6)) 
plt.plot(s.index, s.values, label="Факт") 
plt.plot(yhat.index, yhat_capped.values, label=f"Прогноз 12 (cap={CAP})", linewidth=2) 
plt.title(f"ARIMA С ОГРАНИЧИТЕЛЕМ") 
plt.xlabel("Дата"); plt.ylabel("ФЧ, ЧЕЛ.") 
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6) 
plt.show()

#________END_________


# turnover
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_pacf
import matplotlib.pyplot as plt


#ДЛЯ БАЗОВОЙ ПОДГОТОВКИ МОДЕЛИ

#ЗАВОДИМ ДАННЫЕ
df = pd.read_excel('turnover_drp.xlsx')
colums_list = df.columns.tolist()

df['date']  = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')
df['turnover'] = pd.to_numeric(df['turnover'], errors='coerce')

# СТРОИМ ГРАФИК
df = df.dropna(subset=['date', 'turnover']).set_index('date').asfreq('MS')

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df.index, df['turnover'], label='текучесть', linewidth=3, color = 'green')

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.xticks(rotation=45)

ax.set_title('Текучесть ДРП')
ax.set_xlabel('Месяц'); ax.set_ylabel('Значение текучесть')
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.show()

#стационарность
def test_stationarity(series, title=''):
    print(f"Results of ADF Test on {title}:")
    result = adfuller(series, autolag='AIC')
    print(f"ADF Statistic: {result[0]}")
    print(f"p-value: {result[1]}")
    if result[1] > 0.05:
        print("РЯД НЕ СТАЦИОНАРЕН")
    else:
        print("РЯД СТАЦИОНАРЕН")
    for key, value in result[4].items():
        print(f"Critical Value ({key}): {value}")
    print("\n")

test_stationarity(df['turnover'], 'Counts')

#трейн и тест
# разделение данных на трейн и тест
train_size = int(len(df) * 0.8)

train = df.iloc[:train_size]   
test  = df.iloc[train_size:]   

print("Размер train:", train.shape)
print("Размер test :", test.shape)

#диф
# первое дифференцирование
train_diff1 = train["turnover"].diff().dropna()

# проверка функцией 
print("ADF p-value после 1 diff:", adfuller(train_diff1)[1])
test_stationarity(train_diff1, "Train after 1st diff")

#ACF
plt.figure(figsize=(8,4))
plot_acf(train_diff1, lags=10)
plt.title("ACF для train_diff1")
plt.show()

#PACF
plt.figure(figsize=(8,4))
plot_pacf(train_diff1, lags=10, method="ywm")  # "ywm" – более устойчивая оценка
plt.title("PACF для train_diff1")
plt.show()

#модель
p = 1
d = 1
q = 1

model = ARIMA(train["turnover"], order=(p, d, q))
print('--------------------')
model_fit = model.fit()
print('--------------------')
print(model_fit.summary())


#проверка MAPE
forecast = model_fit.forecast(steps=len(test))
mape = np.mean(np.abs((test["turnover"] - forecast) / test["turnover"])) * 100
print('--------------------')
print(f"MAPE: {mape:.2f}%")

#проверка MAE
y_true = test["turnover"]        
y_pred = forecast            
mae = mean_absolute_error(y_true, y_pred)
print('--------------------')
print(f"MAE: {mae:.2f}")

#печать прогнозных значений
forecast = pd.Series(forecast.values, index=test.index, name="forecast")
print('--------------------')
print (forecast)

plt.figure(figsize=(12,6))
plt.plot(train.index, train["turnover"], label="Обучающая выборка", linewidth=3, color = 'green')
plt.plot(test.index, test["turnover"], label="Тестовая выборка", linewidth=3, color="y")
plt.plot(test.index, forecast, label="Прогноз", linewidth=3, linestyle='--', color="blue")

plt.title("Train/Test vs Forecast (ARIMA)")
plt.xlabel("Дата")
plt.ylabel("Значение текучести")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

forecast = pd.Series(forecast.values, index=test.index, name="forecast")
print (forecast)

