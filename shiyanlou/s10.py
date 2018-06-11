import pandas as pd 
import numpy as np 

def create_data():
    """生成示例数据"""
    data = {"x": ['r', 'g', 'r', 'b', 'g', 'g', 'r', 'r', 'b', 'g', 'g', 'r', 'b', 'b', 'g'],
            "y": ['m', 's', 'l', 's', 'm', 's', 'm', 's', 'm', 'l', 'l', 's', 'm', 'm', 'l'],
            "labels": ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B']}
    data = pd.DataFrame(data)
    return data

def get_P_labels(labels):
    """P(种类)先验概率计算
    """
    labels = list(labels)
    P_lables = {}
    for label in labels:
        P_lables[label] = labels.count(label) / float(len(labels))
    return P_lables

def get_P_fea_lab(P_lables, features, data):
    """P(特征|种类)计算
    """
    P_fea_lab = {}
    train_data = data.iloc[:, 1:].values
    labels = data['labels']

    for each_label in P_lables.keys():
        label_index = [ i for i, label in enumerate(labels) if label == each_label]

        for j in range(len(features)):
            feature_index = [ i for i, feature in enumerate(train_data[:, j]) if feature == features[j]]
            fea_lab_count = len(set(label_index) & set(feature_index))
            key = str(features[j]) + '|' + str(each_label)
            P_fea_lab[key] = fea_lab_count / float(len(label_index))
    
    return P_fea_lab

def classify(data, features):
    labels = data['labels']
    P_lable = get_P_labels(labels)
    P_fea_lab = get_P_fea_lab(P_lable, features, data)

    P = {}
    P_show = {}
    for each_label in P_lable:
        P[each_label] = P_lable[each_label]
        for each_feature in features:
            key = str(each_label) + '|' + str(features)
            P_show[key] = P[each_label] * \
                P_fea_lab[str(each_feature) + '|' + str(each_label)]
            P[each_label] = P[each_label] * \
                P_fea_lab[str(each_feature) + '|' + str(each_label)]

    print(P_show)
    features_label = max(P, key=P.get)
    return features_label


if __name__ == '__main__':
    # 查看示例数据
    data = create_data()
    # print(data)

    # 测试get_P_labels函数
    P_lables = get_P_labels(data['labels'])
    # print(P_lables)

    # 测试get_P_fea_lab函数
    # print(get_P_fea_lab(P_lables, ['r', 'm'], data))

    # 测试分类器
    print(classify(data, ['r', 'm']))

