

#######################################################################################################
## 总健康度计算
#######################################################################################################
# 1.层次分析法
import numpy as np
import pandas as pd

# 计算权重和一致性检验
def ahp_weights(matrix):
    # 归一化矩阵
    norm_matrix = matrix / matrix.sum(axis=0)
    
    # 计算权重（即每行的平均值）
    weights = norm_matrix.mean(axis=1)
    
    # 计算一致性比例 CR
    eig_vals, eig_vecs = np.linalg.eig(matrix)
    max_lambda = max(eig_vals)  # 最大特征值
    n = matrix.shape[0]
    CI = (max_lambda - n) / (n - 1)  # 一致性指标 CI
    RI = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.45]  # 随着矩阵维度变化的随机一致性指标
    CR = CI / RI[n-1]
    
    return weights, CR

# 示例工位比较矩阵（假设9个工位）
A_workstations = np.array([
    [1, 3, 1/2, 2, 1, 5, 3, 1/3, 7],
    [1/3, 1, 1/7, 1/3, 1/5, 2, 1, 1/5, 3],
    [2, 7, 1, 5, 3, 9, 6, 2, 8],
    [1/2, 3, 1/5, 1, 1/3, 5, 2, 1/7, 4],
    [1, 5, 1/3, 3, 1, 7, 4, 2, 6],
    [1/5, 1/2, 1/9, 1/5, 1/7, 1, 1/3, 1/9, 2],
    [1/3, 1, 1/6, 1/2, 1/4, 3, 1, 1/7, 4],
    [3, 5, 1/2, 7, 1/2, 9, 7, 1, 6],
    [1/7, 1/3, 1/8, 1/4, 1/6, 1/2, 1/4, 1/6, 1]
])

# 计算工位权重和一致性检验
weights_workstations, CR_workstations = ahp_weights(A_workstations)

print("工位权重：", weights_workstations)
print("一致性比例（工位比较矩阵）：", CR_workstations)

if CR_workstations < 0.1:
    print("一致性检验通过")
else:
    print("一致性检验未通过，请调整判断矩阵")

# 假设你有每个工位的扭矩、电流、温度的得分（9个工位，每个工位3个准则得分）
data = np.array([
    [0.8, 0.7, 0.6],  # 工位1的扭矩、电流、温度得分
    [0.85, 0.75, 0.65],  # 工位2的得分
    [0.9, 0.8, 0.7],    # 工位3的得分
    [0.75, 0.72, 0.62],  # 工位4的得分
    [0.78, 0.76, 0.68],  # 工位5的得分
    [0.83, 0.74, 0.64],  # 工位6的得分
    [0.88, 0.79, 0.72],  # 工位7的得分
    [0.77, 0.78, 0.69],  # 工位8的得分
    [0.82, 0.71, 0.63],  # 工位9的得分
])

# 假设每个工位的准则权重分别为 [w_torque, w_current, w_temperature]
# 我们需要为每个工位计算得分：每个工位的健康度得分可以通过加权平均计算

# 假设每个工位的准则权重（根据实际情况，可以通过AHP得到）
weights_criteria = np.array([0.5, 0.3, 0.2])  # 扭矩0.5、电流0.3、温度0.2

# 计算每个工位的健康度
health_scores = np.dot(data, weights_criteria)  # 用得分和准则权重计算每个工位的健康度

# 合并工位权重和健康度评分
final_scores = weights_workstations * health_scores  # 将每个工位的健康度乘以工位权重

# 输出最终的健康度评分
df_health = pd.DataFrame({
    '工位': [f'工位{i+1}' for i in range(9)],
    '健康度评分': health_scores,
    '加权健康度评分': final_scores
})

print("\n每个工位的健康度评分及加权健康度评分：")
print(df_health)

# 计算整体设备健康度：对加权健康度评分进行求和
overall_health_score = final_scores.sum()
print("\n设备整体健康度评分：", overall_health_score)






# 2.熵权法
def calculate_entropy_weight(df):
    # Step 1: 标准化数据
    # 假设从第 3 列开始是参数列
    data = df.iloc[:, 2:].values
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)

    # 标准化公式
    norm_data = (data - min_val) / (max_val - min_val)

    # Step 2: 计算比重矩阵 p_ij
    p_ij = norm_data / np.sum(norm_data, axis=0)

    # Step 3: 计算熵值
    epsilon = 1e-10  # 避免计算 ln(0)
    e_i = -np.sum(p_ij * np.log(p_ij + epsilon), axis=0) / np.log(p_ij.shape[0])

    # Step 4: 计算权重
    entropy_weight = (1 - e_i) / np.sum(1 - e_i)

    # Step 5: 计算设备的健康度
    health_score = np.dot(norm_data, entropy_weight)

    # 返回设备健康度
    df['health_score'] = health_score
    return df[['device_id', 'param_time', 'health_score']]

# 假设 df 已经加载好了数据
# 这里执行计算
df_health = calculate_entropy_weight(df)

# 查看结果
print(df_health)
