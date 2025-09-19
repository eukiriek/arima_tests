import numpy as np, pandas as pd, matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

# те же series из блока Prophet
S = 12  # сезонность для месячных

TEST_STEPS = min(12, max(1, len(series)//5))
train, test = series.iloc[:-TEST_STEPS], series.iloc[-TEST_STEPS:]

# Параметры SARIMA — подставьте свои (или из Optuna)
p,d,q = 1,1,1
P,D,Q = 1,1,1

model_tr = SARIMAX(train, order=(p,d,q), seasonal_order=(P,D,Q,S),
                   enforce_stationarity=False, enforce_invertibility=False)
res_tr = model_tr.fit(disp=False)

fc_test = res_tr.get_forecast(steps=len(test)).predicted_mean

def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100

MAE_s  = np.mean(np.abs(test.values - fc_test.values))
MAPE_s = mape(test.values, fc_test.values)
print(f"[SARIMA] MAE={MAE_s:.3f} | MAPE={MAPE_s:.2f}% на последних {len(test)} точках")
print(f"SARIMA orders: ARIMA({p},{d},{q}) × Seasonal({P},{D},{Q},{S})")
print("\nПараметры модели (train fit):")
print(res_tr.params.filter(regex=r'^(ar\.|ma\.|seasonal_ar\.|seasonal_ma\.|sigma2)'))

# график валидации
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="orange")
plt.plot(fc_test.index, fc_test, label="SARIMA Forecast on Test", color="green")
plt.title("SARIMA validation")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

# финальная модель на всем ряде + 12 мес прогноз
model_full = SARIMAX(series, order=(p,d,q), seasonal_order=(P,D,Q,S),
                     enforce_stationarity=False, enforce_invertibility=False)
res_full = model_full.fit(disp=False)

future = res_full.get_forecast(steps=12)
yhat   = future.predicted_mean
ci     = future.conf_int()

print("\n[SARIMA] Прогноз на 12 месяцев:")
print(yhat)

plt.figure(figsize=(12,5))
plt.plot(series.index, series, label="Actual")
plt.plot(yhat.index, yhat, label=f"SARIMA 12m Forecast ({(p,d,q)}×{(P,D,Q,S)})", linewidth=2)
plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="95% CI")
plt.title("SARIMA 12-month forecast")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

