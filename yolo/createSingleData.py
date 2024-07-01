import os
from PIL import Image
from tqdm import tqdm


def create_single_image():
    # 定义单个小图片的大小和大图片的大小
    small_image_size = (90, 90)  # 小图片的大小为 100x100
    big_image_size = (140, 170)  # 大图片的大小为 400x400
    # 图片投喂计数君
    count_index_img = 0
    # 图片所在的目录
    source_directory = 'Arknights/images/train'
    # 遍历每张图片并粘贴到大图片中
    for image_file in os.listdir(source_directory):
        x_offset = 25
        y_offset = 40

        new_image = Image.new('RGB', big_image_size, (0, 0, 0))
        # 打开小图片
        small_image = Image.open(os.path.join(source_directory, image_file))

        # 缩放小图片到指定大小
        small_image = small_image.resize(small_image_size)
        # 将小图片粘贴到大图片中
        new_image.paste(small_image, (x_offset, y_offset))
        # 更新x_offset和y_offset
        x_offset += small_image_size[0]
        y_offset += small_image_size[1]

        parts = image_file.split('_')
        index = parts[1]  # 提取位置信息，示例：5
        name = parts[2]  # 提取作者信息，示例：chen

        labels = []  # 标签列表

        output_image_path = f"singleDataSets/image/resized_{index}_{name}_{count_index_img}.png"
        labels_file_path = f"singleDataSets/label/resized_{index}_{name}_{count_index_img}.txt"
        # 保存合成后的大图片
        new_image.save(output_image_path)

        labels.append((index,
                       0.5,
                       0.55,
                       0.55,
                       0.55))

        with open(labels_file_path, 'w') as f:
            for label in labels:
                f.write(f"{label[0]} {label[1]} {label[2]} {label[3]} {label[4]}\n")
        count_index_img += 1


if __name__ == "__main__":
    create_single_image()
