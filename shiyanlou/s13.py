import numpy as np
import pandas as pd
import math

def create_data():
    """示例数据"""
    data_value = np.array([['long', 'thick', 175, 'no', 'man'], ['short', 'medium', 168, 'no', 'man'],
                     ['short', 'thin', 178, 'yes', 'man'], ['short', 'thick', 172, 'no', 'man'],
                     ['long', 'medium', 163, 'no', 'man'], ['short', 'thick', 180, 'no', 'man'],
                     ['long', 'thick', 173, 'yes', 'man'], ['short', 'thin', 174, 'no', 'man'],
                     ['long', 'thin', 164, 'yes', 'woman'], ['long', 'medium', 158, 'yes', 'woman'],
                     ['long', 'thick', 161, 'yes', 'woman'], ['short', 'thin', 166, 'yes', 'woman'],
                     ['long', 'thin', 158, 'no', 'woman'], ['short', 'medium', 163, 'no', 'woman'],
                     ['long', 'thick', 161, 'yes', 'woman'], ['long', 'thin', 164, 'no', 'woman'],
                     ['short', 'medium', 172, 'yes','woman']] )
    columns = np.array(['hair', 'voice', 'height', 'ear_stud','labels'])
    data=pd.DataFrame(data_value.reshape(17,5),columns=columns)
    return data

def get_Ent(data):
    """获取信息熵"""
    num_sample = len(data)  
    label_counts = {}

    for i in range(num_sample):
        each_data = data.iloc[i, :]
        current_label = each_data['labels']
        if current_label not in label_counts.keys():
            label_counts[current_label] = 0
        label_counts[current_label] += 1

    Ent = 0.0
    for key in label_counts:
        prob = label_counts[key] / num_sample
        Ent -= prob * math.log(prob, 2)

    return Ent

def get_gain(data, base_ent, feature):
    """计算信息增益
    """

    feature_list = data[feature] 
    unique_value = set(feature_list)
    feature_ent = 0.0

    for each_feature in unique_value:
        temp_data = data[data[feature] == each_feature]
        weight = len(temp_data) / len(feature_list)
        temp_ent = weight*get_Ent(temp_data)
        feature_ent += temp_ent

    gain = base_ent - feature_ent
    return gain

def get_splitpoint(data, base_ent, feature):
    """获取连续值特征的划分点"""
    # 将连续值排序并转化为浮点类型
    continues_value = data[feature].sort_values().astype(np.float64)
    
    t_set = []
    for i in range(len(continues_value)-1):   # 得到划分的t的集合
        temp_t = (continues_value[i] + continues_value[i+1]) / 2
        t_set.append(temp_t)

    t_ent = {}
    for each_t in t_set:   # 获取每个花费点t的ent
        temp1_data = data[data[feature].astype(np.float64) > each_t]  # 大于t的数据集
        temp2_data = data[data[feature].astype(np.float64) < each_t]
        weight1 = len(temp1_data)/len(data)
        weight2 = len(temp2_data)/len(data)
        temp_ent = base_ent - weight1*get_Ent(temp1_data)-weight2*get_Ent(temp2_data) # 计算每个划分点的信息增益
        t_ent[each_t] = temp_ent
    # print("t_ent:", t_ent)
    final_t = max(t_ent, key=t_ent.get)
    return final_t

def choice_1(x, t):
    if x > t:
        return ">{}".format(t)
    else:
        return "<{}".format(t)

def choose_feature(data):
    """选择最优划分特征"""
    num_feature = len(data.columns) - 1 # 特征数量
    base_ent = get_Ent(data)

    best_gain = 0.0 # 初始化信息增益
    best_feature = data.columns[0]
    for i in range(num_feature):
        temp_gain = get_gain(data, base_ent, data.columns[i])
        if temp_gain > best_gain:
            best_gain = temp_gain
            best_feature = data.columns[i]
    return best_feature

def create_tree(data):
    """构建决策树"""
    feature_list = data.columns[:-1].tolist()
    label_list = data.iloc[:, -1]
    if len(data['labels'].value_counts()) == 1:
        leaf_node = data['labels'].mode().values
        return leaf_node      # 第一个递归结束条件：所有的类标签完全相同
    if len(feature_list) == 1:
        leaf_node = data['labels'].mode().values
        return leaf_node      # 第二个递归结束条件：用完了所有特征
    best_feature = choose_feature(data)
    tree = {best_feature: {}}
    feat_values = data[best_feature]
    unique_value = set(feat_values)
    for value in unique_value:
        temp_data = data[data[best_feature] == value]
        temp_data = temp_data.drop([best_feature], axis=1)
        tree[best_feature][value] = create_tree(temp_data)
    return tree

def classify(tree, test):
    first_feature = list(tree.keys())[0]
    feature_dict = tree[first_feature]
    labels = test.columns.tolist()
    value = test[first_feature][0]
    for key in feature_dict.keys():
        if value == key:
            if type(feature_dict[key]).__name__ == 'dict':
                class_label = classify(feature_dict[key], test)
            else:
                class_label = feature_dict[key]
    return class_label


























if __name__ == '__main__':
    data = create_data()
    # print(data)

    # 根节点信息熵
    base_ent = get_Ent(data)
    # print(base_ent)

    # hair特征 划分的 信息增益
    # gain = get_gain(data, base_ent, 'hair')
    # print(gain)

    # 测试划分点
    final_t = get_splitpoint(data, base_ent, 'height')
    # print(final_t)

    # 数据预处理：连续值
    deal_data = data.copy()
    deal_data["height"] = pd.Series(map(lambda x:choice_1(int(x), final_t), deal_data['height']))
    # print(deal_data)

    # 测试特征选择
    # print(choose_feature(deal_data))

    tree = create_tree(deal_data)
    # print(tree)
    # {
    #     'height': {
    #         '<168.0': {
    #             'voice': {
    #                 'thin': array(['woman'], dtype=object), 
    #                 'thick': array(['woman'], dtype=object), 
    #                 'medium': {
    #                     'ear_stud': {
    #                         'yes': array(['woman'], dtype=object), 
    #                         'no': array(['man'], dtype=object)
    #                     }
    #                 }
    #             }
    #         }, 
    #         '>168.0': {
    #             'voice': {
    #                 'medium': array(['woman'], dtype=object), 
    #                 'thick': array(['man'], dtype=object), 
    #                 'thin': array(['man'], dtype=object)
    #             }
    #         }
    #     }
    # }

    # 分类
    test = pd.DataFrame({"hair":["long"],"voice":["thin"],"height":[163],"ear_stud":["yes"]})
    test['height'] = pd.Series(map(lambda x:choice_1(int(x), final_t), test['height']))
    # print(test)
    label_pred = classify(tree, test)
    print(label_pred)