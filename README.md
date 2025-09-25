# === 1) Остатки на ТЕСТЕ и выгрузка ===
resid_test = (test["fact_q"] - forecast).rename("residual_test")
resid_test.to_frame().to_csv("residuals_test.csv", encoding="utf-8-sig")
print("Сохранил остатки теста в residuals_test.csv")

# === 2) Диагностика остатков на ТРЕЙНЕ ===
# Внутренние (in-sample) остатки модели на трейне
resid_train = model_fit.resid.rename("residual_train")
resid_train.to_frame().to_csv("residuals_train.csv", encoding="utf-8-sig")

# Ljung–Box (есть ли остаточная автокорреляция?)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb = acorr_ljungbox(resid_train.dropna(), lags=[10, 20], return_df=True)
print("\nLjung–Box (train residuals):")
print(lb[["lb_stat", "lb_pvalue"]])

# Визуально глянуть ACF/PACF остатков (если нужно)
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

plt.figure(figsize=(6,4)); plot_acf(resid_train.dropna(), lags=24); plt.title("ACF остатков (train)"); plt.tight_layout(); plt.show()
plt.figure(figsize=(6,4)); plot_pacf(resid_train.dropna(), lags=24, method="ywm"); plt.title("PACF остатков (train)"); plt.tight_layout(); plt.show()

# === 3) Если в остатках есть структура — смоделируем её ARMA и улучшим прогноз ===
from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from sklearn.metrics import mean_absolute_error

best_aic = np.inf
best_order = None
best_model = None

# Небольшая сетка для ARMA(r,s) на остатках трейна
for r in range(3):          # r = 0,1,2
    for s in range(3):      # s = 0,1,2
        try:
            m = ARIMA(resid_train.dropna(), order=(r, 0, s))
            mf = m.fit()
            if mf.aic < best_aic:
                best_aic = mf.aic
                best_order = (r, 0, s)
                best_model = mf
        except Exception:
            pass

print(f"\nЛучшая ARMA для остатков по AIC: order={best_order}, AIC={best_aic:.2f}")

if best_model is not None:
    # прогноз остатков на горизонт теста
    resid_fc = best_model.forecast(steps=len(test))
    resid_fc = resid_fc.rename("resid_forecast")

    # "Исправленный" прогноз = базовый прогноз + прогноз остатков
    forecast_enh = (forecast + resid_fc).rename("forecast_enhanced")

    # метрики
    mape_enh = np.mean(np.abs((test["fact_q"] - forecast_enh) / test["fact_q"])) * 100
    mae_enh  = mean_absolute_error(test["fact_q"], forecast_enh)
    print(f"MAPE (enhanced): {mape_enh:.2f}%  |  MAE (enhanced): {mae_enh:.2f}")

    # выгрузка вместе
    out = (
        pd.concat(
            [test["fact_q"].rename("y_true"),
             forecast.rename("forecast"),
             resid_test,
             resid_fc,
             forecast_enh],
            axis=1
        )
    )
    out.to_csv("forecast_with_residuals.csv", encoding="utf-8-sig")
    print("Сохранил forecast_with_residuals.csv")

    # сравним графиком
    plt.figure(figsize=(10,5))
    plt.plot(train.index, train["fact_q"], label="Train")
    plt.plot(test.index, test["fact_q"], label="Test")
    plt.plot(test.index, forecast, label="Forecast (baseline)", linestyle="--")
    plt.plot(test.index, forecast_enh, label="Forecast + ARMA(residuals)", linestyle=":")
    plt.title("Базовый прогноз vs с учётом структуры остатков")
    plt.xlabel("Дата"); plt.ylabel("fact_q"); plt.legend(); plt.tight_layout(); plt.show()
