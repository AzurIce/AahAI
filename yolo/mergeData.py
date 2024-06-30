import os
import csv
import re

# 文件夹路径，根据你实际的文件夹路径进行修改
folder_path = 'data'
# 初始化CSV文件的头部和数据列表
csv_data = [['id', 'name', 'index']]
tempName = ''
index = -1
filename_pattern = re.compile(r'char_(\d+)_([^_]+)(_\S+)?\.(jpg|png)')
# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        # 使用正则表达式解析文件名
        match = filename_pattern.match(filename)
        if match:
            image_id = int(match.group(1))
            name = match.group(2)
            if tempName != name:
                index += 1
            tempName = name

            # 将数据添加到csv_data中
            csv_data.append([image_id, name, index])

# 写入CSV文件
csv_filename = 'table/image_data.csv'
with open(csv_filename, 'w', newline='', encoding='utf-8') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)

print(f'CSV文件已生成：{csv_filename}')

