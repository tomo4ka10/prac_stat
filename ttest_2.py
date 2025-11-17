import numpy as np
from scipy.stats import t,nct
import matplotlib.pyplot as plt

# 独立した2標本のt検定において有意水準や検定力，効果量からサンプルサイズを決定するコード
# サンプルサイズ決定について学習する目的

# 両側t検定

# 指定する値
alpha = 0.05 # 有意水準
power_want = 0.8 # 指定する検定力 
d = 0.8 # 効果量
n1_n2 = 1 # 2群のサンプルサイズの比率(n1/n2)


# サンプルサイズ探索
n1 = 2  # 最小値からスタート
while True:
    n2 = int(round(n1 * n1_n2))
    df = n1 + n2 - 2 # 自由度
    delta = d * np.sqrt(n1 * n2 / (n1 + n2)) # 非心パラメータ
    t_a2 = t.ppf(1-alpha/2, df) # 上側alpha/2点
    # 両側検定のPower
    power = 1 - nct.cdf(t_a2, df, delta) + nct.cdf(-t_a2, df, delta)
    if power >= power_want:
        break
    n1 += 1

print(f"n1={n1}, n2={n2}, Power={power:.4f}")

# figure：t分布と非心t分布，有意水準，上側alpha/2点など
x = np.linspace(-4,8,100)
plt.plot(x,t.pdf(x,df),color='red',label='t') # t分布，帰無仮説H0
# 両側t検定
x_t = np.linspace(t_a2, 8, 100)
y_t = t.pdf(x_t,df)
plt.fill_between(x_t, y_t, color='tab:red', ec='gray', alpha=0.2)
x_t = np.linspace(-4, -t_a2, 100)
y_t = t.pdf(x_t,df)
plt.fill_between(x_t, y_t, color='tab:red', ec='gray', alpha=0.2)

plt.plot(x,nct.pdf(x,df,delta),'--',color='blue',label='noncentric-t') # 非心t分布，対立仮説H1
x_nt = np.linspace(-4, t_a2, 100)
y_nt = nct.pdf(x_nt,df,delta)
plt.fill_between(x_nt, y_nt, color='tab:blue', ec='gray', alpha=0.2)

plt.xlim(-4,8)
plt.ylim(bottom=0)
plt.xlabel('x')
plt.ylabel('P(x)')
plt.legend()
plt.title(f"df={df}, t_alpha/2={np.round(t_a2,3)}, (n1,n2)={n1,n2}, d={d}")
plt.show()