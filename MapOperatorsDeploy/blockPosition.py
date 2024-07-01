import csv

# 读取 map.txt 文件
with open('test/map.txt', 'r') as f:
    lines = f.readlines()

# 准备输出的CSV文件
csv_filename = 'test/map_data.csv'
csv_headers = ['x', 'y', 'center_x', 'center_y', 'origin']

# 打开CSV文件以写入数据
with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=csv_headers)
    writer.writeheader()

    # 处理每一行数据
    for line in lines:
        # 示例行: (4, 7) -> (1290.0, 620.0)
        # 去除括号并分割成两部分
        parts = line.strip().replace('(', '').replace(')', '').split(' -> ')

        # 第一部分是坐标 (x, y)
        coordinates = parts[0].split(', ')
        x = int(coordinates[1])
        y = int(coordinates[0])

        # 第二部分是中心点 (center_x, center_y)
        center_point = parts[1].split(', ')
        center_x = float(center_point[0])
        center_y = float(center_point[1])

        # 创建原始行数据字符串
        origin = line.strip()

        # 写入CSV文件
        writer.writerow({
            'x': x,
            'y': y,
            'center_x': center_x,
            'center_y': center_y,
            'origin': origin
        })

print(f'已成功生成CSV文件 "{csv_filename}"。')