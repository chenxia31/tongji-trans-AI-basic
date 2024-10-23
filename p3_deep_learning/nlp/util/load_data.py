def load_sentence_polarity(data_path,train_ratio=0.8):
    """
    data_path:输入数据的位置
    train_ratio:训练集和测试集的比例划分
    """
    all_data=[]
    categories=[]
    with open(data_path,'r',encoding='utf8') as file:
        for sample in file.readlines():
            # polar是类别
            # 0 positive
            # 1 negative
            polar,sent=sample.strip().split('\t')
            categories.append(polar)
            all_data.append((polar,sent))
    length=len(all_data)
    train_len=int(length*train_ratio)
    train_data=all_data[:train_len]
    test_data=all_data[train_len:]
    return train_data,test_data,categories



