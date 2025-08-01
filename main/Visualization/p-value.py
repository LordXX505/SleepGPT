import numpy as np
import scipy.stats as stats

# 模型的均值和标准差
mean_model_1 = 87.4
std_model_1 = 0.4
mean_model_2 = 86.5
std_model_2 = 0.8

# 实验次数
n1 = 5
n2 = 5

# 计算 t 值
t_stat = (mean_model_1 - mean_model_2) / np.sqrt((std_model_1**2 / n1) + (std_model_2**2 / n2))

# 自由度 df
df = n1 + n2 - 2

# 计算 p 值
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

print(f"t 值: {t_stat}, p 值: {p_value}")