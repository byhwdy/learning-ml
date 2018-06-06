import pandas as pd 

def create_data():
    """生成示例数据"""
    data = {"x": ['r', 'g', 'r', 'b', 'g', 'g', 'r', 'r', 'b', 'g', 'g', 'r', 'b', 'b', 'g'],
            "y": ['m', 's', 'l', 's', 'm', 's', 'm', 's', 'm', 'l', 'l', 's', 'm', 'm', 'l'],
            "labels": ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B']}
    data = pd.DataFrame(data)
    return data



if __name__ == '__main__':
    # 查看示例数据
    data = create_data()
    print(data)