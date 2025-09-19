# !pip install prophet --quiet

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from prophet import Prophet

# ==== данные ====
df["DATE"]  = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
df = df.dropna(subset=["DATE","TOTAL"]).sort_values("DATE")

# приводим к регулярной месячной частоте (если нужно — смените на 'D'/'Q')
series = (df.set_index("DATE")["TOTAL"]
            .asfreq(pd.infer_freq(df["DATE"]) or "MS")
            .interpolate())

# Prophet-формат
df_p = series.reset_index().rename(columns={"DATE":"ds","TOTAL":"y"})

# ==== валидационная разбивка ====
TEST_STEPS = min(12, max(1, len(df_p)//5))
train_p = df_p.iloc[:-TEST_STEPS].copy()
test_p  = df_p.iloc[-TEST_STEPS:].copy()

# ==== модель Prophet ====
# Если показатель в ДОЛЯХ (0..1), часто лучше multiplicative:
model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                seasonality_mode="multiplicative")
# при необходимости можно добавить регрессоры/праздники:
# model.add_regressor("X1"); ...

model.fit(train_p)

# прогноз на длину теста
future_test = model.make_future_dataframe(periods=TEST_STEPS, freq=series.index.freq)
fc_test_p   = model.predict(future_test).iloc[-TEST_STEPS:][["ds","yhat"]].set_index("ds")["yhat"]

# метрики
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100

MAE_p  = np.mean(np.abs(test_p.set_index("ds")["y"] - fc_test_p))
MAPE_p = mape(test_p.set_index("ds")["y"], fc_test_p)
print(f"[Prophet] MAE={MAE_p:.3f} | MAPE={MAPE_p:.2f}% на последних {TEST_STEPS} точках")

# график валидации
plt.figure(figsize=(12,5))
plt.plot(train_p["ds"], train_p["y"], label="Train")
plt.plot(test_p["ds"], test_p["y"], label="Test", color="orange")
plt.plot(fc_test_p.index, fc_test_p.values, label="Prophet Forecast on Test", color="green")
plt.title("Prophet validation")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

# ==== финальный прогноз 12 мес на полном ряде ====
model_full = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                     seasonality_mode="multiplicative")
model_full.fit(df_p)

future12 = model_full.make_future_dataframe(periods=12, freq=series.index.freq)
fc12_p   = model_full.predict(future12).iloc[-12:][["ds","yhat","yhat_lower","yhat_upper"]].set_index("ds")

print("\n[Prophet] Прогноз на 12 месяцев:")
print(fc12_p)

plt.figure(figsize=(12,5))
plt.plot(series.index, series.values, label="Actual")
plt.plot(fc12_p.index, fc12_p["yhat"], label="Prophet 12m Forecast", linewidth=2)
plt.fill_between(fc12_p.index, fc12_p["yhat_lower"], fc12_p["yhat_upper"], alpha=0.2, label="Interval")
plt.title("Prophet 12-month forecast")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

