import pandas as pd
import numpy as np

# ========= 1) 长表 -> 宽表 =========
def long_to_wide(df, value = 'param_score', time_col='param_time'):
    """
    将长表(df: device_id, param_mark, param_value, param_time[ms])转成宽表：
    每行 = 某设备在某时刻的所有参数 + 时间列
    """
    df = df.copy()
    # 排序：先时间后param_mark（可选）
    df = df.sort_values([time_col, 'param_mark'])

    # 透视
    wide = df.pivot_table(
        index=['device_id', time_col],
        columns='param_mark',
        values= value,
        aggfunc='last',    # 若同一时刻重复上报，取最后一条
    ).reset_index()

    # # 显式列顺序（如果你只要 plcparam1~27）
    # if param_marks is None:
    #     param_marks = [f'plcparam{i}' for i in range(1, 28)]
    # cols = ['device_id', time_col] + param_marks
    # # 可能有的列暂时不存在，reindex会自动补NaN
    # wide = wide.reindex(columns=cols)

    # 过滤掉那些只有一个参数的时间点
    df_wide_filtered = wide.dropna(how='any', subset=[col for col in wide.columns if col not in ['device_id', 'param_time']])
    return df_wide_filtered


# ========= 2) 添加时间特征 =========
def add_time_features(df_wide, time_col='param_time', unit='ms', tz='UTC'):
    """
    将毫秒(或秒)级Unix时间戳转成时间特征：
    - dt（带时区）
    - timestamp_s（连续时间秒）
    - Δt（秒/分钟）
    - 小时/星期/月的sin-cos编码
    """
    out = df_wide.copy()

    # 时间戳 -> pandas 时间
    dt_utc = pd.to_datetime(out[time_col], unit=unit, utc=True)
    if tz and tz.upper() != 'UTC':
        dt_local = dt_utc.dt.tz_convert(tz)
    else:
        dt_local = dt_utc

    out['dt'] = dt_local
    # 连续数值时间（秒），供模型学习全局趋势/先后
    out['timestamp_s'] = dt_utc.view('int64') / 1e9

    # 周期性编码
    hour = out['dt'].dt.hour
    dow  = out['dt'].dt.dayofweek            # Monday=0
    mon  = out['dt'].dt.month - 1            # 0~11

    out['hour_sin'] = np.sin(2*np.pi*hour/24)
    out['hour_cos'] = np.cos(2*np.pi*hour/24)
    out['dow_sin']  = np.sin(2*np.pi*dow/7)
    out['dow_cos']  = np.cos(2*np.pi*dow/7)
    out['mon_sin']  = np.sin(2*np.pi*mon/12)
    out['mon_cos']  = np.cos(2*np.pi*mon/12)

    # Δt：同一设备相邻两条的时间差（秒、分钟）
    out = out.sort_values(['device_id', 'dt'])
    out['delta_t_s'] = out.groupby('device_id')['timestamp_s'].diff().fillna(0.0)
    out['delta_t_min'] = out['delta_t_s'] / 60.0

    return out


# ========= 3) 缺失值处理（前向填充+可选插值/填0）=========
def impute_features(df, param_marks, method='ffill', fillna_value=None):
    """
    对传感器列做缺失值处理：
    - method='ffill': 按device_id分组前向填充
    - fillna_value=0: 再把剩余NaN统一填0（可选）
    """
    out = df.copy()
    out[param_marks] = (
        out.groupby('device_id')[param_marks]
        .apply(lambda g: g.ffill() if method=='ffill' else g)
        .reset_index(level=0, drop=True)
    )
    if fillna_value is not None:
        out[param_marks] = out[param_marks].fillna(fillna_value)
    return out


# ========= 4) 滑动窗口构造 =========
def make_windows(
    df_feat,
    feature_cols,
    target_cols,
    window_size=60,
    horizon=1,
    stride=1,
    by_device=True,
    drop_incomplete=True
):
    """
    将特征表转成监督学习样本(X, y)：
    - X 形状: [num_samples, window_size, num_features]
    - y 形状: [num_samples, len(target_cols)] 或 [num_samples, horizon, len(target_cols)]（你可以自行改成序列预测）
    参数：
      - window_size: 用过去多少步作为输入窗口
      - horizon: 预测步长（T+horizon时刻的目标）
      - stride: 滑窗步进
      - by_device: 是否在设备内部分别滑窗（强烈建议True）
      - drop_incomplete: 是否丢弃不完整窗口
    """
    X_list, y_list, meta = [], [], []

    def _one_device_make(g):
        g = g.sort_values('dt')
        vals = g[feature_cols].values
        tars = g[target_cols].values
        times = g['dt'].values

        n = len(g)
        # 预测的目标点索引：i + window_size - 1 + horizon
        for start in range(0, n - window_size - horizon + 1, stride):
            end = start + window_size
            tgt_idx = end - 1 + horizon
            if tgt_idx >= n:
                if drop_incomplete:
                    break
                else:
                    continue
            X_list.append(vals[start:end])
            y_list.append(tars[tgt_idx])
            # 记录窗口对应的设备、起止时间
            meta.append({
                'device_id': g['device_id'].iloc[0],
                'start_time': times[start],
                'end_time': times[end-1],
                'target_time': times[tgt_idx],
            })

    if by_device:
        for dev, g in df_feat.groupby('device_id'):
            _one_device_make(g)
    else:
        _one_device_make(df_feat)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta)
    return X, y, meta


# ========= 5) 一个把所有步骤串起来的pipeline =========
def build_pdmpipeline(
    df_long,
    param_marks=None,
    time_col='param_time',
    unit='ms',
    tz='UTC',
    impute_method='ffill',
    impute_fillna=None,
    window_size=60,
    horizon=1,
    stride=1,
    target_cols=('plcparam1',),  # 你要预测的列（可多列）
    extra_feature_cols=None      # 额外想带入的列名（如自定义统计特征）
):
    # 默认参数列
    if param_marks is None:
        param_marks = [f'plcparam{i}' for i in range(1, 28)]

    # 1) 长->宽
    wide = long_to_wide(df_long, param_marks=param_marks, time_col=time_col)

    # 2) 时间特征
    feat = add_time_features(wide, time_col=time_col, unit=unit, tz=tz)

    # 3) 缺失处理
    feat = impute_features(feat, param_marks, method=impute_method, fillna_value=impute_fillna)

    # 4) 选择特征列
    time_feat_cols = [
        'timestamp_s', 'delta_t_s', 'delta_t_min',
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'mon_sin', 'mon_cos'
    ]
    base_feature_cols = list(param_marks) + time_feat_cols
    if extra_feature_cols:
        base_feature_cols += list(extra_feature_cols)
        base_feature_cols = list(dict.fromkeys(base_feature_cols))  # 去重保持顺序

    # 5) 构造滑动窗口
    X, y, meta = make_windows(
        feat,
        feature_cols=base_feature_cols,
        target_cols=list(target_cols),
        window_size=window_size,
        horizon=horizon,
        stride=stride,
        by_device=True,
        drop_incomplete=True
    )

    return {
        'features_df': feat,                # 含时间与参数的特征表(行=设备时刻)
        'feature_cols': base_feature_cols,  # 用到的特征列名
        'target_cols': list(target_cols),   # 目标列名
        'X': X,                             # [N, T, F]
        'y': y,                             # [N, C]
        'meta': meta                        # 每个样本的设备/时间映射
    }




# 假设你的原始df包含四列：
# df.columns = ['device_id', 'param_mark', 'param_value', 'param_time']
# 其中 param_time 是毫秒级Unix时间戳（如 1753164785521）

result = build_pdmpipeline(
    df_long=df,
    param_marks=[f'plcparam{i}' for i in range(1, 28)],
    time_col='param_time',
    unit='ms',                  # 如果是秒，改为 's'
    tz='Asia/Shanghai',         # 业务时区（或 'UTC'）
    impute_method='ffill',      # 缺失前向填充（同设备）
    impute_fillna=None,         # 仍缺的值保留 NaN（你也可以设为0）
    window_size=60,             # 窗口长度（例如过去60个采样点）
    horizon=1,                  # 预测1步后的目标
    stride=1,                   # 每步滑动1个点
    target_cols=('plcparam1',), # 例子：预测 plcparam1（可多列）
    extra_feature_cols=None
)

X = result['X']          # 形状: [样本数, 60, 特征数]
y = result['y']          # 形状: [样本数, 目标维度]
meta = result['meta']    # 每个样本对应的设备/时间
print(X.shape, y.shape)
print(meta.head())

