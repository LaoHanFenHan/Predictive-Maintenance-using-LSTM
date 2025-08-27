#未来状态预测（Multi-step Forecasting）
#任务形式：序列预测，预测未来 k 步的关键参数（温度、压力、振动等）。

#优点

#不依赖明确的失效标注，只需要设备历史运行数据。

#能通过“未来参数是否异常”间接推断故障风险。

#可以作为“虚拟传感器”使用（缺失数据补偿、异常趋势识别）。

#缺点

#需要结合专家知识/阈值来定义“异常”。

#如果未来预测偏差积累，长时预测容易失真。

#适用

#设备运行记录全，但维护/故障标注少的情况。

#想做“数字孪生”（预测未来趋势，与仿真对比）。


# 一、关键超参（改这里就能切换任务）
WINDOW_SIZE = 60          # 用过去多少个时间步做输入
HORIZONS = (1, 3, 5, 10)  # 要同时预测未来哪些步（T+1、T+3、T+5、T+10）
TARGET_COLS = ['plcparam1', 'plcparam2']  # 同时预测哪些参数（可只放一个）
TZ = 'UTC'                # 或你的业务时区 'Asia/Shanghai'
UNIT = 'ms'               # 你的时间戳单位（毫秒）

#二、从长表到特征表
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Dropout, Dense

# ---------- 1) 长 -> 宽 ----------
def long_to_wide(df, time_col='param_time', param_col='param_mark', value_col='param_value'):
    df = df.sort_values([time_col, param_col])
    wide = df.pivot_table(index=['device_id', time_col],
                          columns=param_col, values=value_col,
                          aggfunc='last').reset_index()
    # 显式列顺序（按需调整/扩展）
    param_marks = [f'plcparam{i}' for i in range(1, 28)]
    cols = ['device_id', time_col] + param_marks
    wide = wide.reindex(columns=cols)
    return wide

# ---------- 2) 加时间特征 ----------
def add_time_features(df_wide, time_col='param_time', unit='ms', tz='UTC'):
    out = df_wide.copy()
    dt_utc = pd.to_datetime(out[time_col], unit=unit, utc=True)
    dt_local = dt_utc if (tz is None or tz.upper()=='UTC') else dt_utc.dt.tz_convert(tz)
    out['dt'] = dt_local
    out['timestamp_s'] = dt_utc.view('int64') / 1e9

    hour = out['dt'].dt.hour
    dow  = out['dt'].dt.dayofweek
    mon  = out['dt'].dt.month - 1
    out['hour_sin'] = np.sin(2*np.pi*hour/24); out['hour_cos'] = np.cos(2*np.pi*hour/24)
    out['dow_sin']  = np.sin(2*np.pi*dow/7);   out['dow_cos']  = np.cos(2*np.pi*dow/7)
    out['mon_sin']  = np.sin(2*np.pi*mon/12);  out['mon_cos']  = np.cos(2*np.pi*mon/12)

    out = out.sort_values(['device_id', 'dt'])
    out['delta_t_s']   = out.groupby('device_id')['timestamp_s'].diff().fillna(0.0)
    out['delta_t_min'] = out['delta_t_s'] / 60.0
    return out

# ---------- 3) 缺失处理 ----------
def impute_by_device_ffill(df, sensor_cols, fill0=False):
    out = df.copy()
    out[sensor_cols] = (
        out.groupby('device_id')[sensor_cols]
        .apply(lambda g: g.ffill())
        .reset_index(level=0, drop=True)
    )
    if fill0:
        out[sensor_cols] = out[sensor_cols].fillna(0)
    return out


#三、构造“多步×多目标”滑动窗口
def make_windows_multi_forecast(
    df_feat, feature_cols, target_cols, window_size=60, horizons=(1,3,5), stride=1
):
    """
    返回：
      X      : [N, T, F]
      y_reg  : [N, H, C]  (H=len(horizons), C=len(target_cols))
      meta   : 样本元信息
    """
    X_list, y_list, meta = [], [], []
    H, C = len(horizons), len(target_cols)

    for dev, g in df_feat.groupby('device_id'):
        g = g.sort_values('dt')
        F = g[feature_cols].values
        T = g[target_cols].values
        n = len(g)

        for start in range(0, n - window_size - max(horizons) + 1, stride):
            end = start + window_size
            # 采集不同 horizon 的目标
            y_steps = []
            ok = True
            for h in horizons:
                idx = end - 1 + h
                if idx >= n:
                    ok = False; break
                y_steps.append(T[idx])  # [C]
            if not ok: continue

            X_list.append(F[start:end])                # [T, F]
            y_list.append(np.stack(y_steps, axis=0))   # [H, C]
            meta.append({'device_id': dev, 'start': start, 'end': end-1})

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, pd.DataFrame(meta)


#四、模型：共享编码器 + 多步多目标回归头
#简洁高效：一次输出 H*C 个值，训练后再 reshape 回 [H, C] 做评估。
def build_forecaster(input_len, feature_dim, H, C, hidden1=128, hidden2=64, dropout=0.2):
    x_in = Input(shape=(input_len, feature_dim))
    x = LSTM(hidden1, return_sequences=True)(x_in)
    x = Dropout(dropout)(x)
    x = LSTM(hidden2, return_sequences=False)(x)
    x = Dropout(dropout)(x)
    out = Dense(H * C, name='regression')(x)  # 线性输出
    model = Model(inputs=x_in, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


###main
# === 载入你的原始长表 df ===
# df.columns = ['device_id','param_mark','param_value','param_time']
# df = pd.read_csv('your_data.csv')

# 1) 长->宽
wide = long_to_wide(df)

# 2) 时间特征 + 缺失处理
sensor_cols = [f'plcparam{i}' for i in range(1, 28)]
feat = add_time_features(wide, time_col='param_time', unit=UNIT, tz=TZ)
feat = impute_by_device_ffill(feat, sensor_cols, fill0=False)

# 3) 选择特征列（传感器 + 时间）
time_cols = ['timestamp_s','delta_t_s','delta_t_min','hour_sin','hour_cos','dow_sin','dow_cos','mon_sin','mon_cos']
feature_cols = sensor_cols + time_cols

# 4) 构造多步窗口
X, y, meta = make_windows_multi_forecast(
    feat, feature_cols, TARGET_COLS, window_size=WINDOW_SIZE, horizons=HORIZONS, stride=1
)
N, T, F = X.shape
H, C = y.shape[1], y.shape[2]
print("X:", X.shape, "y:", y.shape)  # 例如 (n, 60, F), (n, 4, 2)

# 5) 归一化（注意避免泄漏：严谨做法是先时间/设备切分，再在训练集fit）
X2d = X.reshape(-1, F)
x_scaler = MinMaxScaler().fit(X2d)        # 演示：全量fit；生产请仅在训练集fit
X = x_scaler.transform(X2d).reshape(N, T, F)

y2d = y.reshape(N, -1)                    # [N, H*C]
y_scaler = MinMaxScaler().fit(y2d)        # 多目标一起缩放
y2d = y_scaler.transform(y2d)

# 6) 切分集（演示随机；生产建议按时间或设备留后）
X_train, X_test, y_train, y_test = train_test_split(X, y2d, test_size=0.2, random_state=42)

# 7) 建模训练
model = build_forecaster(WINDOW_SIZE, F, H, C)
model.summary()
history = model.fit(
    X_train, y_train,
    epochs=80, batch_size=256, validation_split=0.1, verbose=2
)

# 8) 评估 + 反归一化 + 按 horizon/目标拆解误差
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}  MAE: {mae:.4f}")

y_pred_2d = model.predict(X_test, verbose=0)
y_pred = y_scaler.inverse_transform(y_pred_2d).reshape(-1, H, C)
y_true = y_scaler.inverse_transform(y_test).reshape(-1, H, C)

# 按 horizon 统计 MAE / MAPE（示例）
for hi, h in enumerate(HORIZONS):
    mae_h = np.mean(np.abs(y_pred[:, hi, :] - y_true[:, hi, :]), axis=0)
    print(f"H+{h} steps ahead MAE per target:", dict(zip(TARGET_COLS, mae_h)))


