import os
import glob

# 设置您要清理的文件夹的顶层路径
base_folder_path = '/Users/nibabi/Desktop/skateboard_trick_classification/Tricks'

# 设置您要删除的文件的格式
file_extension = '.npy'

# 使用glob模块递归找到所有的 .mov 文件
files_to_delete = glob.glob(os.path.join(base_folder_path, '**', '*' + file_extension), recursive=True)

# 遍历找到的文件列表并删除每个文件
for file_path in files_to_delete:
    try:
        os.remove(file_path)
        print(f"Deleted file: {file_path}")
    except OSError as e:
        print(f"Error deleting file {file_path}: {e.strerror}")

print("All files with the specified format have been deleted from the folder and its subfolders.")
