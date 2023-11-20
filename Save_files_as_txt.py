import os

# 定义文件夹路径
folder_path = '/Users/nibabi/Desktop/skateboard_trick_classification/Tricks/normal_ollie' 

# 获取文件夹中所有.mp4文件的路径
mp4_file_paths = [os.path.join(root, file) for root, dirs, files in os.walk(folder_path) for file in files if file.endswith('.mp4')]

# 定义要保存的文本文件名
output_file = 'Test_ollie.txt'  # 可以自定义文件名

# 将.mp4文件路径写入文本文件
with open(output_file, 'w') as f:
    for mp4_file_path in mp4_file_paths:
        f.write(mp4_file_path + '\n')

with open('/Users/nibabi/Desktop/skateboard_trick_classification/Test_list.txt') as f:
    test_list = [row.strip() for row in f]

# 提取每个路径的最后两个部分
last_two_parts = [os.path.basename(os.path.dirname(path)) + '/' + os.path.basename(path) for path in test_list]

for part in last_two_parts:
    print(part)
