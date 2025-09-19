import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# ==== 0) данные ====
# df: колонки ["DATE","TOTAL"]
df["DATE"]  = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
df = df.dropna(subset=["DATE","TOTAL"]).sort_values("DATE").set_index("DATE")

# регулярная частота (важно!)
freq = pd.infer_freq(df.index) or "MS"     # месяцы с начала месяца
srs  = df["TOTAL"].asfreq(freq).interpolate()

# ==== 1) параметры SARIMA (пример) ====
p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 12     # s=12 для месячных

# ==== 2) train/test split ====
TEST_STEPS = min(12, len(srs)//5 or 1)
train, test = srs.iloc[:-TEST_STEPS], srs.iloc[-TEST_STEPS:]

# ==== 3) обучение на train ====
model_tr = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,s),
                   enforce_stationarity=False, enforce_invertibility=False)
res_tr = model_tr.fit(disp=False)

# ==== 4) прогноз на длину test и метрики ====
fc_test = res_tr.get_forecast(steps=len(test)).predicted_mean

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100

MAE  = np.mean(np.abs(test.values - fc_test.values))
MAPE = mape(test.values, fc_test.values)

print(f"Validation metrics on last {len(test)} points:")
print(f"  MAE  = {MAE:.3f}")
print(f"  MAPE = {MAPE:.2f}%")

# ==== 5) вывод pdqs и коэффициентов ====
print(f"\nSelected orders: ARIMA(p,d,q)=({p},{d},{q}), Seasonal(P,D,Q,s)=({P},{D},{Q},{s})")
# основные оценённые коэфы
print("\nEstimated params (train fit):")
print(res_tr.params.filter(regex=r'^(ar\.|ma\.|seasonal_ar\.|seasonal_ma\.|sigma2)'))

# (График валидации)
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="orange")
plt.plot(fc_test.index, fc_test, label="Forecast on Test", color="green")
plt.title("SARIMA validation")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

# ==== 6) финальная модель на всём ряду + прогноз 12 мес ====
model_full = SARIMAX(srs, order=(p,d,q), seasonal_order=(P,D,Q,s),
                     enforce_stationarity=False, enforce_invertibility=False)
res_full = model_full.fit(disp=False)
fc12 = res_full.get_forecast(steps=12).predicted_mean

print("\n12-month ahead forecast:")
print(fc12)

plt.figure(figsize=(12,5))
plt.plot(srs.index, srs, label="Actual")
plt.plot(fc12.index, fc12, label="12-month Forecast", linewidth=2)
plt.title(f"SARIMA{(p,d,q)}×{(P,D,Q,s)}")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()
