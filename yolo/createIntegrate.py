import os
from PIL import Image

# 定义单个小图片的大小和大图片的大小
small_image_size = (90, 90)  # 小图片的大小为 100x100
big_image_size = (360, 360)    # 大图片的大小为 400x400

# 图片所在的目录
directory = 'ArkNights/images/train/'

# 保存大图片和标签信息
output_image_path = 'extraDatasets/image/output_big_image.png'
# 输出标签信息到文件
labels_file_path = 'extraDatasets/label/labels.txt'


info = []  # 单一图片信息
labels = []  # 标签列表

# 图片投喂计数君
count_index_img = 0

# 准备小图片列表
small_images = []
for filename in os.listdir(directory):
    if filename.endswith('.png') and count_index_img < 16:
        # 构建完整的文件路径
        full_path = os.path.join(directory, filename)
        # 解析和提取信息
        parts = filename.split('_')
        id_number = parts[0].split('.')[0]  # 提取编号，例如：000010
        index = parts[1]  # 提取位置信息，示例：5
        name = parts[2]  # 提取作者信息，示例：chen
        count = parts[3].split('.')[0]  # 提取序号，示例：13

        # 判断条件：count除以10余1才执行操作
        if int(count) % 10 == 1:
            # 打开图片
            img = Image.open(full_path)
            img = img.resize(small_image_size)  # 调整大小为指定的小图片大小
            small_images.append(img)

            # 记录标签信息
            info.append({
                'count_index_img': count_index_img,
                'id_number': id_number,
                'index': index,
                'name': name,
                'count': count,
            })

            count_index_img += 1  # 增加执行次数

# 创建新的大图片和对应的标签数据
big_image = Image.new('RGBA', big_image_size, (0, 0, 0))  # 创建白色背景的大图片
#
# 在大图片中放置小图片，并记录中心点位置
offset_x = 0
offset_y = 0
for idx, img in enumerate(small_images):
    big_image.paste(img, (offset_x, offset_y))
    # 计算中心点位置
    center_x = offset_x + small_image_size[0] // 2
    center_y = offset_y + small_image_size[1] // 2

    # 查找对应的info信息
    matched_info = None
    for item in info:
        if item['count_index_img'] == idx:
            matched_info = item
            break

    if matched_info is not None:
        labels.append((matched_info['index'],
                       center_x / big_image_size[0],
                       center_y / big_image_size[1],
                       0.250000,
                       0.250000))
    else:
        labels.append(("image", idx, center_x, center_y, "failure"))

    # 更新放置位置，这里简单地按顺序横向排列
    offset_x += small_image_size[0]
    if offset_x >= big_image_size[0]:
        offset_x = 0
        offset_y += small_image_size[1]
#

big_image.save(output_image_path)  # 保存大图片
print(f"保存大图片成功！路径：{output_image_path}")
#

with open(labels_file_path, 'w') as f:
    for label in labels:
        f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")

print(f"保存标签信息成功！路径：{labels_file_path}")
