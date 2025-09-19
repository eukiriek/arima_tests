# ПАРАМЕТРЫ МОДЕЛИ
p, d, q = 1, 1, 1          # несезонные порядки
P, D, Q, s = 1, 1, 1, 12   # сезонные порядки и период (12 = месяцы)

# (опционально) ограничение прогноза
USE_CAP = True
CAP = 1.0                  # например, если моделируете доли (0..1). Для штук поставьте свой порог.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import STL

# ==== ПОДГОТОВКА ДАННЫХ ====
# Ожидаем, что у вас есть df с колонками DATE, TOTAL
df["DATE"]  = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
df = df.dropna(subset=["DATE","TOTAL"]).sort_values("DATE").set_index("DATE")

# приводим к регулярной частоте (если не распознана — задаём вручную)
freq = pd.infer_freq(df.index)
if freq is None:
    freq = "MS"          # начало месяца; если нужен конец месяца — "M"
srs = df["TOTAL"].asfreq(freq).interpolate()   # мягко закрываем единичные пропуски

# (необязательно) быстрый взгляд на сезонность
# STL(srs, period=s).fit().plot(); plt.show()

# ==== ОБУЧЕНИЕ SARIMA НА ВСЕМ РЯДЕ ====
model = SARIMAX(
    srs,
    order=(p, d, q),
    seasonal_order=(P, D, Q, s),
    enforce_stationarity=False,
    enforce_invertibility=False
)
res = model.fit(disp=False)
print(res.summary())

# ==== ПРОГНОЗ НА 12 МЕСЯЦЕВ ====
steps = 12
fc = res.get_forecast(steps=steps)
yhat = fc.predicted_mean.copy()  # прогноз
ci   = fc.conf_int()             # доверительные интервалы

# ---- (опционально) ограничение потолком ----
if USE_CAP:
    # если за первые 6 месяцев достиг потолка — дальше держим плоско на CAP
    HALF_YEAR = 6
    reached = (yhat >= CAP).values
    yhat_cap = yhat.copy()

    if reached.any():
        first_hit = int(np.argmax(reached))
        if first_hit < HALF_YEAR:
            yhat_cap.iloc[first_hit:] = CAP
        else:
            yhat_cap = yhat_cap.clip(upper=CAP)
    else:
        yhat_cap = yhat_cap.clip(upper=CAP)

    # ограничим и интервалы
    ci.iloc[:, 0] = np.minimum(ci.iloc[:, 0].values, CAP)
    ci.iloc[:, 1] = np.minimum(ci.iloc[:, 1].values, CAP)

    yhat = yhat_cap

print("\nПрогноз на 12 месяцев:")
print(yhat)

# ==== ВИЗУАЛИЗАЦИЯ ====
plt.figure(figsize=(12,6))
plt.plot(srs.index, srs, label="Факт")
plt.plot(yhat.index, yhat, label=f"Прогноз 12 мес (SARIMA{(p,d,q)}×{(P,D,Q,s)})", linewidth=2)
plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="95% CI")
plt.title("SARIMA прогноз на 12 месяцев")
plt.xlabel("Месяц"); plt.ylabel("TOTAL")
plt.legend(); plt.grid(True, linestyle="--", alpha=0.6)
plt.show()

# (опционально) сохранить прогноз
# out = pd.DataFrame({"actual": srs, "forecast": yhat})
# out.to_csv("sarima_forecast_12m.csv", encoding="utf-8-sig")

