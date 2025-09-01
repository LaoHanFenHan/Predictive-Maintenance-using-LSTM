

#######################################################################################################
## 总健康度计算
#######################################################################################################
# 1.层次分析法
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
    print("工位权重：", weights)
    print("一致性比例（工位比较矩阵）：", CR)

    if CR < 0.1:
        print("一致性检验通过")
    else:
        print("一致性检验未通过，请调整判断矩阵")
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




# 假设每个工位的准则权重（根据实际情况，可以通过AHP得到）
weights_criteria = np.array([0.5, 0.3, 0.2])  # 扭矩0.5、电流0.3、温度0.2
# 创建一个包含27个参数的权重数组
param_weights = np.array([
    [0.5] * 9,  # 扭矩对应的9个参数
    [0.3] * 9,  # 电流对应的9个参数
    [0.2] * 9   # 温度对应的9个参数
]).flatten()
# 获取所有参数列名，去掉 'device_id' 和 'param_time' 列
param_columns = [col for col in df_wide.columns if col not in ['device_id', 'param_time']]
# 提取所有的参数列
data = df_wide[param_columns]

# 计算每个工位的健康度
health_scores = np.dot(data, param_weights)  # 用得分和准则权重计算每个工位的健康度

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
