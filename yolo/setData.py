import os
import shutil
import re

# 文件夹路径，根据你实际的文件夹路径进行修改
folder_path = 'data'
destination_path = 'ArkNights/images/train'
tempName = ''
index = -1
filename_pattern = re.compile(r'char_(\d+)_([^_]+)(_\S+)?\.(jpg|png)')
counter = 1
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

            image_name = f"{formatted_id}_{index}_{name}_{counter}.png"
            print(image_name)

            # 构建源文件完整路径
            source_file = os.path.join(folder_path, filename)

            # 构建目标文件完整路径
            destination_file = os.path.join(destination_path, image_name)

            # 复制文件
            shutil.copy(source_file, destination_file)
            print(f"Copied {filename} to {destination_file} : {counter} done.")

            counter += 1

