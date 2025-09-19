# ==== БИБЛИОТЕКИ ====
import warnings, numpy as np, pandas as pd, optuna
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
warnings.filterwarnings("ignore")

# ==== 0) ДАННЫЕ ====
# Ожидаем df с колонками: DATE, TOTAL
# df = pd.read_excel("test.xlsx")
df["DATE"]  = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
df["TOTAL"] = pd.to_numeric(df["TOTAL"], errors="coerce")
df = df.dropna(subset=["DATE","TOTAL"]).sort_values("DATE").set_index("DATE")

# Приводим к регулярной частоте (важно для прогноза)
freq = pd.infer_freq(df.index)
if freq is None:
    freq = "MS"  # месяцы с начала месяца; для кварталов: "QS" / "Q", для дней: "D"
series = df["TOTAL"].asfreq(freq).interpolate()

# ==== 1) НАСТРОЙКИ ВАЛИДАЦИИ ====
S = 12                         # сезонный период (12 для месячных)
TEST_STEPS = min(12, max(1, len(series)//5))  # тест — последние точки (обычно 12)
train, test = series.iloc[:-TEST_STEPS], series.iloc[-TEST_STEPS:]

# Метрика MAPE
def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    eps = 1e-9
    return np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), eps, None))) * 100

# ==== 2) ЦЕЛЕВАЯ ФУНКЦИЯ ДЛЯ OPTUNA ====
def objective(trial):
    # Диапазоны разумные и не слишком широкие (устойчивость подбора)
    p = trial.suggest_int("p", 0, 3)
    d = trial.suggest_int("d", 0, 2)
    q = trial.suggest_int("q", 0, 3)

    P = trial.suggest_int("P", 0, 2)
    D = trial.suggest_int("D", 0, 1)
    Q = trial.suggest_int("Q", 0, 2)

    try:
        model = SARIMAX(
            train,
            order=(p, d, q),
            seasonal_order=(P, D, Q, S),
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        res = model.fit(disp=False)

        # прогноз на длину теста и метрика
        fc = res.get_forecast(steps=len(test)).predicted_mean
        score = mape(test.values, fc.values)

        # лёгкий «анти-переобучительный» штраф за сложность модели
        complexity = (p+q) + (P+Q) + d + 2*D
        return score + 0.05 * complexity
    except Exception:
        # неустойчивая модель — большой штраф
        return 1e9

# ==== 3) ЗАПУСК OPTUNA ====
sampler = optuna.samplers.TPESampler(seed=42)
pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
study   = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)
study.optimize(objective, n_trials=60, show_progress_bar=True)

print("Best value (val MAPE + penalty):", f"{study.best_value:.2f}")
print("Best params:", study.best_params)

# ==== 4) ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ НА ТЕСТЕ (чистые метрики) ====
bp = study.best_params
best_model = SARIMAX(
    train,
    order=(bp["p"], bp["d"], bp["q"]),
    seasonal_order=(bp["P"], bp["D"], bp["Q"], S),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

fc_test = best_model.get_forecast(steps=len(test)).predicted_mean
MAE  = np.mean(np.abs(test.values - fc_test.values))
MAPE = mape(test.values, fc_test.values)
print(f"\nValidation on last {len(test)} points | MAE={MAE:.3f} | MAPE={MAPE:.2f}%")
print(f"Selected orders: ARIMA({bp['p']},{bp['d']},{bp['q']}) × Seasonal({bp['P']},{bp['D']},{bp['Q']},{S})")

# Валидационный график
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Test", color="orange")
plt.plot(fc_test.index, fc_test, label="Forecast on Test", color="green")
plt.title("SARIMA validation")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

# ==== 5) ФИНАЛ: ПЕРЕОБУЧЕНИЕ НА ВСЁМ РЯДЕ И ПРОГНОЗ 12 ====
final = SARIMAX(
    series,
    order=(bp["p"], bp["d"], bp["q"]),
    seasonal_order=(bp["P"], bp["D"], bp["Q"], S),
    enforce_stationarity=False,
    enforce_invertibility=False
).fit(disp=False)

future = final.get_forecast(steps=12)
yhat   = future.predicted_mean
ci     = future.conf_int()

print("\n12-month ahead forecast:")
print(yhat)

plt.figure(figsize=(12,5))
plt.plot(series.index, series, label="Actual")
plt.plot(yhat.index, yhat, label=f"12m Forecast (SARIMA{(bp['p'],bp['d'],bp['q'])}×{(bp['P'],bp['D'],bp['Q'],S)})", linewidth=2)
plt.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:,1], alpha=0.2, label="95% CI")
plt.title("Best SARIMA forecast (Optuna)")
plt.grid(True, linestyle="--", alpha=0.6); plt.legend(); plt.show()

