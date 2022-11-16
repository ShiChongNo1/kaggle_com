import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder, MinMaxScaler


def difference(dataset, interval=1):
    """
    create difference list
    :param dataset:
    :param interval: start
    :return: diff list
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return diff


def inverse_difference(last_ob, value):
    """
    :param last_ob: origin data
    :param value: diff data
    :return:
    """
    return value + last_ob


# 将数据缩放到 [-1, 1]之间的数
def scale(train):
    # 创建一个缩放器
    # print(train.shape)
    train.where(torch.isnan(train), torch.full_like(train, 0))
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # 将train从二维数组的格式转化为一个23*2的张量
    #train = train.reshape(train.shape[0], train.shape[1])
    # 使用缩放器将数据缩放到[-1, 1]之间
    train_scaled = scaler.transform(train)
    return torch.Tensor(train_scaled)


# 数据逆缩放，scaler是之前生成的缩放器，X是一维数组，y是数值
def invert_scale(scaler, X, y):
    # 将x，y转成一个list列表[x,y]->[0.26733207, -0.025524002]
    # [y]可以将一个数值转化成一个单元素列表
    new_row = [x for x in X] + [y]
    #new_row = [X[0]]+[y]
    # 将列表转化为一个,包含两个元素的一维数组，形状为(2,)->[0.26733207 -0.025524002]
    array = np.array(new_row)
    print(array.shape)
    # 将一维数组重构成形状为(1,2)的，1行、每行2个元素的，2维数组->[[ 0.26733207 -0.025524002]]
    array = array.reshape(1, len(array))
    # 逆缩放输入的形状为(1,2),输出形状为(1,2) -> [[ 73 15]]
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


def fixna(data: pd.DataFrame):
    data.fillna(method='pad', inplace=True)
    data.fillna(method='backfill', inplace=True)


def data_mapping(data):
    object_cols = [cname for cname in data.columns
                   if data[cname].dtype == "object"
                   and cname != "date"]
    ordinal_encoder = OrdinalEncoder()
    data[object_cols] = ordinal_encoder.fit_transform(data[object_cols])
    return object_cols, data


def label_encoder(data: pd.DataFrame, columns: list):
    le = LabelEncoder()
    for col in columns:
        data[col] = le.fit_transform(data[col])


def format_data(dataset, transaction, holidays, oil, store):
    # 清理商店无用信息
    valid_extra_store_inf = store.drop(axis=1, columns=['type', 'cluster'])

    # 合并得到带有位置信息的训练集
    data_state = pd.merge(dataset, valid_extra_store_inf,
                          how='left', left_on='store_nbr', right_on='store_nbr')

    # 清理无效假日，删除无用数据
    valid_holiday = holidays.drop(holidays[holidays['transferred']].index)
    valid_holiday.drop(
        axis=1, columns=['locale', 'description', 'transferred'], inplace=True)

    # 合并 带有位置信息的训练集 与 地区假日信息 为 标准训练集
    valid_holiday.rename(
        columns={'locale_name': 'city', 'type': 'events'}, inplace=True)
    format_datas = pd.merge(data_state, valid_holiday, how='left', left_on=[
                            'date', 'city'], right_on=['date', 'city'])
    valid_holiday.rename(columns={'city': 'state'}, inplace=True)
    format_datas = pd.merge(format_datas, valid_holiday, how='left', left_on=['date', 'state'],
                            right_on=['date', 'state'])

    # 处理标准训练集假期数据
    event_nan = format_datas[format_datas['events_x'].isna()]
    event_not_nan = format_datas[format_datas['events_x'].notna()]

    # - 拼接假期数据
    event_nan['events_x'].fillna(value='', inplace=True)
    event_nan['events_y'].fillna(value='', inplace=True)
    event_nan['events_x'] = event_nan['events_x'] + event_nan['events_y']
    format_datas = pd.concat([event_nan, event_not_nan], axis=0).sort_index()
    format_datas.drop(
        axis=1, columns=['city', 'state', 'events_y'], inplace=True)
    format_datas.rename(columns={'events_x': 'events'}, inplace=True)

    # 处理标准训练集无效数据
    fixna(format_datas)
    # format_datas.dropna(inplace=True)

    # 拼接油价数据
    fixna(oil)
    format_datas = pd.merge(format_datas, oil, how='left',
                            left_on='date', right_on='date')
    fixna(format_datas)

    # 数据编码
    columns = ['family', 'events']
    label_encoder(format_datas, columns)

    # 返回标准dataset
    return format_datas

  