import numpy as np
from scipy.stats import t,nct
import matplotlib.pyplot as plt

# 1標本のt検定において有意水準や検定力，効果量からサンプルサイズを決定するコード
# 母平均に差があるかないか検定をする際のサンプルサイズ決定について学習する目的で作った

def samplesize_ttest1_means(alpha, power_want, d, tail_test):
    # サンプルサイズ探索
    n = 1  # 最小値からスタート
    while True:
        df = n - 1 # 自由度
        delta = d * np.sqrt(n) # 非心パラメータ
        
        # 両側or片側検定
        match tail_test:
            case 'two-tailed': # 両側検定
                t_a = t.ppf(1-alpha/2, df) # alpha/2点
                power = 1 - nct.cdf(t_a, df, delta) + nct.cdf(-t_a, df, delta)
            case 'upper-one-tailed': # 上側片側検定
                t_a = t.ppf(1-alpha, df) # alpha点
                power = 1 - nct.cdf(t_a, df, delta)
            case 'lower-one-tailed': # 下側片側検定
                t_a = t.ppf(alpha, df) # alpha点
                delta = -1*delta
                power = nct.cdf(t_a, df, delta)

        # 検出力が設定した検定力より大きくなるまでサンプルサイズを探索
        if power >= power_want:
            break
        n += 1

    return n, df, delta, t_a, power

def show_samplesize_ttest1_means(df, delta, t_a, tail_test):
    display_x = np.array([-8,8]) # x軸のlim

    fig, ax = plt.subplots()

    # t分布，非心t分布を描画
    x = np.linspace(display_x[0], display_x[1],100)
    plt.plot(x,t.pdf(x,df),color='red',label='t') # t分布，帰無仮説H0
    plt.plot(x,nct.pdf(x,df,delta),'--',color='blue',label='noncentric-t') # 非心t分布，対立仮説H1

    # 両側or片側検定でt分布, 非心t分布の塗りつぶす範囲について場合分け
    match tail_test:
        case 'two-tailed': # 両側検定
            # t分布
            x_t = np.concatenate([np.linspace(display_x[0], -t_a, 200),[np.nan],np.linspace(t_a, display_x[1], 200)]) # 左右を連結
            y_t = t.pdf(x_t,df)
            # 非心t分布
            x_nt = np.linspace(display_x[0], t_a, 100)
            y_nt = nct.pdf(x_nt,df,delta)

        case 'upper-one-tailed': # 上側片側検定
            # t分布
            x_t = np.linspace(t_a, display_x[1], 100)
            y_t = t.pdf(x_t,df)
            # 非心t分布
            x_nt = np.linspace(display_x[0], t_a, 100)
            y_nt = nct.pdf(x_nt,df,delta)

        case 'lower-one-tailed': # 下側片側検定
            # t分布
            x_t = np.linspace(display_x[0], t_a, 100)
            y_t = t.pdf(x_t,df)
            # 非心t分布
            x_nt = np.linspace(t_a, display_x[1], 100)
            y_nt = nct.pdf(x_nt,df,delta)    

    # 塗りつぶし
    plt.fill_between(x_t, y_t, color='tab:red', ec='gray', alpha=0.2)
    plt.fill_between(x_nt, y_nt, color='tab:blue', ec='gray', alpha=0.2)

    # 棄却域における%点の図示
    ax.axvline(x=t_a, label='t_a point')

    # 表示の設定
    plt.xlim(display_x[0],display_x[1])
    plt.ylim(bottom=0)
    plt.xlabel('x')
    plt.ylabel('P(x)')
    plt.legend()
    plt.title(f"t_alpha={np.round(t_a,3)}")
    plt.show()

# ----実行----
# 指定する値
alpha = 0.05 # 有意水準
power_want = 0.8 # 指定する検定力 
d = 0.5 # 効果量(Cohen's d)：平均の差を標本標準偏差で割ったもの
tail_test = 'two-tailed' # 検定方法（two-tailed/upper-one-tailed/lower-one-tailed）

[n, df, delta, t_a, power] = samplesize_ttest1_means(alpha, power_want, d, tail_test)
show_samplesize_ttest1_means(df, delta, t_a, tail_test)
print(f"n={n}, Actual Power={power:.4f}, Critical t:{t_a:.4f}, δ:{delta:.4f}")
