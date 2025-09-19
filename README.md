import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf

# допустим у вас DataFrame df с индексом-датами и колонкой "turnover"
series = df["turnover"]

# 1. Декомпозиция классическим методом (seasonal_decompose)
res = seasonal_decompose(series, model="additive", period=12)  # 12 месяцев
res.plot()
plt.suptitle("Seasonal Decomposition (Additive)", fontsize=14)
plt.show()

# 2. Более гибкая декомпозиция STL
stl = STL(series, period=12)
res_stl = stl.fit()
res_stl.plot()
plt.suptitle("STL Decomposition", fontsize=14)
plt.show()

# 3. ACF для поиска сезонных лагов
plot_acf(series, lags=36)
plt.title("ACF для проверки сезонности")
plt.show()

