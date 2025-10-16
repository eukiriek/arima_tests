# === 1) Остатки на обучении (in-sample) ===
yhat_train = linreg.predict(t_train)
residuals_train = train['value'].values - yhat_train

# Печать и быстрая диагностика остатков
print("\n=== Residuals (train) summary ===")
print(pd.Series(residuals_train, index=train.index).describe())
print("Mean residual (bias):", residuals_train.mean())

# === 2) Bias-коррекция прогноза ===
# Если средний остаток != 0, добавим его к прогнозу, чтобы убрать систематическое смещение.
lr_pred_bias_corrected = lr_pred + residuals_train.mean()

# === 3) (Опционально) Сезонная коррекция остатков по месяцам ===
# Работает корректно, если у ряда месячная частота (MS/M).
# Идея: если регрессия по времени не учитывает сезонность, она «останется» в остатках.
try:
    # средний остаток по месяцам (1..12)
    resid_by_month = (
        pd.DataFrame({'resid': residuals_train}, index=train.index)
        .groupby(train.index.month)['resid']
        .mean()
        .reindex(range(1,13))  # на всякий случай выровняем на все месяцы
    )
    # сопоставим каждому прогнозируемому месяцу его средний остаток
    seasonal_resid_for_test = pd.Index(test.index.month).map(resid_by_month.get).to_numpy()
    lr_pred_seasonal_corrected = lr_pred + seasonal_resid_for_test

    # Комбинированная коррекция: и bias, и сезонный профиль остатков
    lr_pred_bias_seasonal = lr_pred + residuals_train.mean() + seasonal_resid_for_test
except Exception as e:
    print("Seasonal residual correction skipped:", e)
    resid_by_month = None
    lr_pred_seasonal_corrected = None
    lr_pred_bias_seasonal = None

# === 4) Оценка качества (на тесте) для разных вариантов ===
def eval_model(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mp = mape(y_true.values, y_pred)
    print(f"{name} -> MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mp:.2f}%")

print("\n=== Evaluation on test ===")
eval_model("Linear Regression (raw)", test['value'], lr_pred)
eval_model("Linear Regression + bias", test['value'], lr_pred_bias_corrected)

if lr_pred_seasonal_corrected is not None:
    eval_model("Linear Regression + seasonal residuals", test['value'], lr_pred_seasonal_corrected)
    eval_model("Linear Regression + bias + seasonal", test['value'], lr_pred_bias_seasonal)

# === 5) Таблица с прогнозами и корректировками ===
out = pd.DataFrame({
    'actual': test['value'],
    'lr_pred_raw': lr_pred,
    'lr_pred_bias': lr_pred_bias_corrected
}, index=test.index)

if lr_pred_seasonal_corrected is not None:
    out['lr_pred_seasonal'] = lr_pred_seasonal_corrected
    out['lr_pred_bias_seasonal'] = lr_pred_bias_seasonal
    out['seasonal_resid_used'] = seasonal_resid_for_test

print("\n=== Forecast table (head) ===")
print(out.head())

# === 6) Графики: исходный прогноз и корректировки ===
plt.figure(figsize=(10,4))
plt.plot(train.index, train['value'], label='train')
plt.plot(test.index,  test['value'],  label='test', linewidth=2)

plt.plot(test.index,  lr_pred,                 '--', label='LR raw')
plt.plot(test.index,  lr_pred_bias_corrected,  '--', label='LR + bias')

if lr_pred_seasonal_corrected is not None:
    plt.plot(test.index,  lr_pred_seasonal_corrected, '--', label='LR + seasonal')
    plt.plot(test.index,  lr_pred_bias_seasonal,      '--', label='LR + bias + seasonal')

plt.title(f'Линейная регрессия: корректировка прогнозов остатками (h={h})')
plt.legend()
plt.tight_layout()
plt.show()

# === 7) Диагностика остатков: визуально и численно ===
# В идеале остатки должны быть «белым шумом»: без тренда/сезонности и автокорреляции.
plt.figure(figsize=(10,3))
pd.Series(residuals_train, index=train.index).plot()
plt.title("Residuals (train)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,3))
pd.Series(residuals_train).hist(bins=20)
plt.title("Residuals distribution (train)")
plt.tight_layout()
plt.show()
