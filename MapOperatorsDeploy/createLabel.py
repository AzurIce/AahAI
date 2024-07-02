import os
import pandas as pd


# 文件夹路径，根据你实际的文件夹路径进行修改
folder_path = 'test'
image_path = folder_path + '/image'
label_path = folder_path + '/label'
counter = 1

# 读取 CSV 文件到 DataFrame
df = pd.read_csv('test/map_data.csv')

for filename in os.listdir(image_path):
    if filename.endswith('.png'):
        # 分割文件名
        parts = filename.split('_')
        index = parts[0]
        name = parts[1]
        position_x = parts[2]
        position_y = parts[3]
        direction = parts[4].split('.')[0]

        # 取得放置地块的地图信息
        filtered_row = df.loc[(df['x'] == int(position_x)) & (df['y'] == int(position_y))]
        center_x_value = filtered_row['center_x'].values[0]
        center_y_value = filtered_row['center_y'].values[0]

        index_x = center_x_value / 1920.0
        index_y = center_y_value / 1080.0 - 0.025 * (int(position_y) * 0.1 + 1)

        # 构建目标文件完整路径
        label_filename = filename.split('.')[0] + '.txt'
        result_file_path = os.path.join(label_path, label_filename)
        #
        with open(result_file_path, 'w') as f:
            line = f"{index} {str(index_x)} {str(index_y)} 0.06 0.1475\n"
            f.write(line)

        print(f"{counter} labels done")
        counter += 1
