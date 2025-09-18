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



новый

import numpy as np

# ---- параметры ограничения ----
CAP = 1000          # <<< ваш порог (константа)
HALF_YEAR = 6       # полгода (в шагах прогноза)

# базовый прогноз
yhat = fc.predicted_mean.copy()
ci = fc.conf_int()

# ---- логика "упёрся в константу" ----
reached = (yhat >= CAP).values
yhat_capped = yhat.copy()

if reached.any():
    first_hit_pos = int(np.argmax(reached))  # позиция первого превышения
    if first_hit_pos < HALF_YEAR:
        # если достигли порога в первые 6 месяцев — дальше плоско на CAP
        yhat_capped.iloc[first_hit_pos:] = CAP
    else:
        # если порог достигнут позже 6 мес — просто не даём превысить CAP
        yhat_capped = yhat_capped.clip(upper=CAP)
else:
    # порог в горизонте не достигнут — на всякий случай не превышаем порог
    yhat_capped = yhat_capped.clip(upper=CAP)

# ---- печать и график с ограничением ----
print("\nПрогноз (c ограничением):")
print(yhat_capped)

plt.figure(figsize=(12,6))
plt.plot(s.index, s.values, label="Факт")
plt.plot(yhat.index, yhat_capped.values, label=f"Прогноз 12 (cap={CAP})", linewidth=2)
plt.fill_between(ci.index, ci.iloc[:,0].clip(upper=CAP), ci.iloc[:,1].clip(upper=CAP),
                 alpha=0.2, label="95% CI (с cap)")
plt.title(f"ARIMA({p},{d},{q}) — прогноз с ограничением на уровень {CAP}")
plt.xlabel("Дата"); plt.ylabel("TOTAL")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.show()
