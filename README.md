# ваш блок
p = 1
d = 3
q = 1

from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

# ---- ДОБАВЛЕНО: подготовка ряда с датами и частотой ----
s = pd.to_numeric(df["TOTAL"], errors="coerce").dropna()
s.index = pd.to_datetime(df.index)              # если уже datetime — ок
freq = pd.infer_freq(s.index)
if freq is None:
    freq = "MS"                                 # поменяйте при другой периодичности: "M","D","Q" и т.д.
s = s.asfreq(freq).interpolate()                # выровнять и закрыть единичные пропуски

# ---- модель на всём ряду ----
model = ARIMA(s, order=(p, d, q),
              enforce_stationarity=False,
              enforce_invertibility=False)
model_fit = model.fit()

print('----------------')
print(model_fit.summary())

# ---- прогноз на 12 шагов вперёд ----
fc = model_fit.get_forecast(steps=12)
yhat = fc.predicted_mean        # значения прогноза
ci = fc.conf_int()              # доверительные интервалы

print("\nПрогноз на 12 периодов:")
print(yhat)

# ---- график факт + прогноз ----
plt.figure(figsize=(12,6))
plt.plot(s.index, s.values, label="Факт")
plt.plot(yhat.index, yhat.values, label="Прогноз (12)", linewidth=2)
plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="95% CI")
plt.title("ARIMA(1,3,1) — прогноз на 12 месяцев")
plt.xlabel("Дата"); plt.ylabel("TOTAL")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# (опционально) сохранить в CSV
# pd.DataFrame({"actual": s, "forecast": yhat}).to_csv("forecast_12.csv", encoding="utf-8-sig")
