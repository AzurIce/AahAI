import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

def draw_boxes_on_image(image_path, data_path, output_path):
    # 打开图像
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # 读取数据文件
    with open(data_path, 'r') as file:
        lines = file.readlines()
    
    # 解析并绘制每个框
    for line in lines:
        values = line.strip().split()
        box_id = int(values[0])
        x_center = float(values[1])
        y_center = float(values[2])
        box_width = float(values[3])
        box_height = float(values[4])
        
        # 计算框的坐标
        left = (x_center - box_width / 2) * width
        right = (x_center + box_width / 2) * width
        top = (y_center - box_height / 2) * height
        bottom = (y_center + box_height / 2) * height
        
        # 绘制矩形框
        draw.rectangle([left, top, right, bottom], outline="red", width=2)
        draw.text((left, top), str(box_id), fill="red")
    
    # 保存带有框的图像
    image.save(output_path)
    print(f"Image with boxes saved to {output_path}")

# 示例调用
id = 600
draw_boxes_on_image(f"./datasets/arknights/images/train/{id}.png", f"./datasets/arknights/labels/train/{id}.txt", "./output.png")
