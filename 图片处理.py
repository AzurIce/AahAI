from PIL import Image


def resize_image(input_image_path, output_image_path, size):
    original_image = Image.open(input_image_path)
    resized_image = original_image.resize(size)
    resized_image.save(output_image_path)


input_image = 'resources/input/operators_det/1.png'  # 输入图片的路径
output_image = 'resources/input/operators_det/2.png'  # 输出图片的路径
size = (640, 640)  # 目标尺寸

resize_image(input_image, output_image, size)

print('图片处理完成')
