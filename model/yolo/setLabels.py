import os
import shutil
import re

# 文件夹路径，根据你实际的文件夹路径进行修改
folder_path = 'data'
destination_path = 'ArkNights/labels/train'
filename_pattern = re.compile(r'char_(\d+)_([^_]+)(_\S+)?\.(jpg|png)')
counter = 1
tempName = ''
index = -1
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 使用正则表达式解析文件名
        match = filename_pattern.match(filename)
        if match:
            image_id = int(match.group(1))
            formatted_id = f"{image_id:06}"     # 补全为6位字符串
            name = match.group(2)

            if tempName != name:
                index += 1
            tempName = name

            label_name = f"{formatted_id}_{index}_{name}_{counter}.txt"
            print(label_name)

            # 构建目标文件完整路径
            result_file_path = os.path.join(destination_path, label_name)

            with open(result_file_path, 'w') as f:
                line = f"{index} 0.500000 0.500000 0.800000 0.990000\n"
                f.write(line)

            counter += 1
