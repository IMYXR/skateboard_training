import os
import random
from sklearn.model_selection import train_test_split

# 路径配置
base_folder = '/Users/nibabi/Desktop/skateboard_trick_classification/Tricks'
train_txt = 'train.txt'
dev_txt = 'dev.txt'
test_txt = 'test.txt'

# 数据划分比例：训练集，开发集和测试集
train_ratio = 0.7
dev_ratio = 0.2
# 测试集比例自动计算为剩余部分

# 收集所有文件的相对路径和标签
data = []
for label_folder in os.listdir(base_folder):
    folder_path = os.path.join(base_folder, label_folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith(('.mov')):  # 根据需要更改文件类型
                relative_path = os.path.join(label_folder, file)  # 保存相对路径
                data.append(relative_path)

# 数据划分
train_dev_data, test_data = train_test_split(data, test_size=1 - train_ratio - dev_ratio, random_state=42)
train_data, dev_data = train_test_split(train_dev_data, test_size=dev_ratio / (train_ratio + dev_ratio), random_state=42)

# 将数据写入到txt文件
def write_to_txt(file_list, file_name):
    with open(file_name, 'w') as f:
        for item in file_list:
            f.write(f'{item}\n')

write_to_txt(train_data, train_txt)
write_to_txt(dev_data, dev_txt)
write_to_txt(test_data, test_txt)

print("数据划分完成，文件路径已保存。")
