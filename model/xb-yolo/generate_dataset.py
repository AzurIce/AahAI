from concurrent.futures import ProcessPoolExecutor
from PIL import Image, ImageSequence
import os
import random
from tqdm import tqdm
import numpy as np

operators = [
    "char_285_medic2", # ⭐ 医疗小车
    "char_502_nblade", # ⭐⭐ 夜刀
    "char_500_noirc",  # ⭐⭐ 黑角
    "char_503_rang",   # ⭐⭐ 巡林者
    "char_501_durin",  # ⭐⭐ 杜林
    "char_009_12fce",  # ⭐⭐ 12F
    "char_284_spot",   # ⭐⭐⭐ 斑点
    "char_281_popka",  # ⭐⭐⭐ 泡普卡
    "char_283_midn",   # ⭐⭐⭐ 月见夜
    "char_282_catap",  # ⭐⭐⭐ 空爆
    "char_278_orchid", # ⭐⭐⭐ 梓兰
    "char_210_stward", # ⭐⭐⭐ 史都华德
    "char_212_ansel",  # ⭐⭐⭐ 安塞尔
    "char_120_hibisc", # ⭐⭐⭐ 芙蓉
    "char_121_lava",   # ⭐⭐⭐ 炎熔
    "char_211_adnach", # ⭐⭐⭐ 安德切尔
    "char_124_kroos",  # ⭐⭐⭐ 克洛丝
    "char_122_beagle", # ⭐⭐⭐ 米格鲁
    "char_209_ardign", # ⭐⭐⭐ 卡提
    "char_208_melan",  # ⭐⭐⭐ 玫兰莎
    "char_192_falco",  # ⭐⭐⭐ 翎羽
    "char_240_wyvern", # ⭐⭐⭐ 香草
    "char_123_fang",   # ⭐⭐⭐ 芬
    "char_151_myrtle", # ⭐⭐⭐⭐ 桃金娘
]
# 24 / 8 = 3

SCALE_FACTOR = 0.5

# 接收 background 和 img，返回生成的 imgs
def overlay_img_on_background(background_path, img_path, num=5):
    background = Image.open(background_path)
    img = Image.open(img_path).convert('RGBA')
    
    bg_w, bg_h = background.width, background.height
    img_w, img_h = img.width, img.height
    img_w, img_h = int(img_w * SCALE_FACTOR), int(img_h * SCALE_FACTOR)
    img = img.resize((img_w, img_h))
    
    res = []
    random.seed(0)
    for _ in range(num):
        pixel_x = random.random()
        pixel_y = random.random()
        pixel_x = int(pixel_x * (bg_w - img_w))
        pixel_y = int(pixel_y * (bg_h - img_h))
        
        generated_img = background.copy()
        generated_img.paste(img, (pixel_x, pixel_y), img)
        
        # label x, y, w, h
        x = (pixel_x + img_w // 2) / bg_w
        y = (pixel_y + img_h // 2) / bg_h
        w, h = img_w / bg_w, img_h / bg_h
        
        res.append((f'{x} {y} {w} {h}', generated_img))
    return res
    

def generate_unique_pairs(n, x_range, y_range):
    pairs = set()
    while len(pairs) < n:
        x = random.randint(*x_range)
        y = random.randint(*y_range)
        pairs.add((x, y))
    return list(pairs)


def get_operator_images(operator):
    res = []

    for facing in ['front', 'back']:
        res_dir = f'./exports/{facing}/{operator}'
        files = os.listdir(res_dir)
        res += [os.path.join(res_dir, f) for f in files if f.split('_')[0] in animation]

    return res

boxes = [[(264.0, 98.0),
(403.0, 98.0),
(542.0, 98.0),
(682.0, 98.0),
(821.0, 98.0),
(960.0, 98.0),
(1099.0, 98.0),
(1238.0, 98.0),
(1378.0, 98.0),
(1517.0, 98.0),
(1656.0, 98.0),
],
[(230.0, 202.0),
(376.0, 202.0),
(522.0, 202.0),
(668.0, 202.0),
(814.0, 202.0),
(960.0, 202.0),
(1106.0, 202.0),
(1252.0, 202.0),
(1398.0, 202.0),
(1544.0, 202.0),
(1690.0, 202.0),
],
[(218.0, 355.0),
(367.0, 355.0),
(515.0, 355.0),
(663.0, 355.0),
(812.0, 355.0),
(960.0, 355.0),
(1108.0, 355.0),
(1257.0, 355.0),
(1405.0, 355.0),
(1553.0, 355.0),
(1702.0, 355.0),
],
[(179.0, 481.0),
(335.0, 481.0),
(492.0, 481.0),
(648.0, 481.0),
(804.0, 481.0),
(960.0, 481.0),
(1116.0, 481.0),
(1272.0, 481.0),
(1428.0, 481.0),
(1585.0, 481.0),
(1741.0, 481.0),
],
[(136.0, 620.0),
(301.0, 620.0),
(466.0, 620.0),
(630.0, 620.0),
(795.0, 620.0),
(960.0, 620.0),
(1125.0, 620.0),
(1290.0, 620.0),
(1454.0, 620.0),
(1619.0, 620.0),
(1784.0, 620.0),
],
[(51.0, 750.0),
(232.0, 750.0),
(414.0, 750.0),
(596.0, 750.0),
(778.0, 750.0),
(960.0, 750.0),
(1142.0, 750.0),
(1324.0, 750.0),
(1506.0, 750.0),
(1688.0, 750.0),
(1869.0, 750.0),
],
[(-9.0, 931.0),
(185.0, 931.0),
(379.0, 931.0),
(572.0, 931.0),
(766.0, 931.0),
(960.0, 931.0),
(1154.0, 931.0),
(1348.0, 931.0),
(1541.0, 931.0),
(1735.0, 931.0),
(1929.0, 931.0),
]]

def find_bounding_box(image):
    # 打开图像并转换为RGBA格式
    width, height = image.size
    pixels = image.load()

    # 初始化边界
    left, right, top, bottom = width, 0, height, 0

    # 遍历图像，查找非透明像素的边界
    for y in range(height):
        for x in range(width):
            _, _, _, a = pixels[x, y]
            if a != 0:  # 如果像素不是透明的
                if x < left:
                    left = x
                if x > right:
                    right = x
                if y < top:
                    top = y
                if y > bottom:
                    bottom = y

    # 如果没有找到非透明像素，返回None
    if left == width and right == 0 and top == height and bottom == 0:
        return None

    # 返回边界
    return left, top, right, bottom

def put_operator_img_on_image(img_path, background, pos):
    img = Image.open(img_path).convert('RGBA')
    
    bg_w, bg_h = background.width, background.height
    img_w, img_h = img.width, img.height
    img_w, img_h = int(img_w * SCALE_FACTOR), int(img_h * SCALE_FACTOR)
    img = img.resize((img_w, img_h))
    
    x, y = pos
    # pixel_x = int(x * (bg_w - img_w))
    # pixel_y = int(y * (bg_h - img_h))
    pixel_x, pixel_y = boxes[int(y)][int(x)]
    pixel_x, pixel_y = int(pixel_x), int(pixel_y)
    
    l, t, r, b = find_bounding_box(img)
    nw = r - l
    nh = b - t
    mw = (l + r) // 2
    mh = (t + b) // 2

    generated_img = background.copy()
    generated_img.paste(img, (pixel_x - mw, pixel_y - mh), img)
    
    # label x, y, w, h
    # MAX_W = 130
    # MAX_H = 200
    # img_w = min(MAX_W, img_w)
    # img_h = min(MAX_H, img_h)
    x = pixel_x / bg_w
    y = pixel_y / bg_h
    w, h = nw / bg_w, nh / bg_h
    
    return generated_img, f'{x} {y} {w} {h}'

MAX_ROW = 6  # [0, 6]
MAX_COL = 10 # [0, 10]
dataset_name = 'arknights'

if __name__ == '__main__':
    random.seed(0)
    animation = ['Attack', 'Default', 'Idle']
    facing_list = ['front', 'back']
    
    images_dir = f'./datasets/{dataset_name}/images/train'
    labels_dir = f'./datasets/{dataset_name}/labels/train'
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    background = Image.open("background.png")

    # 随机坐标 (topleft)
    pos_list = generate_unique_pairs(8, (1, MAX_COL - 1), (1, MAX_ROW - 1))
    # pos_list = [(x / MAX_COL, y / MAX_ROW) for x, y in pos_list]
    cnt = 0
    for squad_idx, squad in enumerate(np.array_split(operators, 3)):
        for k in range(1):

            images = [background.copy()]
            labels = [""]
            # 将干员的每一帧依次放在 imgs 中，如果不够则用最后一张
            for i, operator in tqdm(list(enumerate(squad))):
                operator_images = get_operator_images(operator)
                while len(operator_images) > len(images):
                    images.append(images[-1].copy())
                    labels.append(labels[-1])
                for j, oper_img in enumerate(operator_images):
                    img, rect = put_operator_img_on_image(oper_img, images[j], pos_list[i])
                    images[j] = img
                    labels[j] += f'{i + squad_idx * 8} {rect}\n'
            for image, label in tqdm(list(zip(images, labels)), desc='saving'):
                image.save(os.path.join(images_dir, f'{cnt}.png'), format='PNG')
                with open(os.path.join(labels_dir, f'{cnt}.txt'), 'w') as f:
                    f.write(label)
                # exit(0)
                cnt += 1
    classes = [f'  {idx}: {tag}\n' for idx, tag in enumerate(operators)]
    with open(f'./datasets/{dataset_name}/{dataset_name}.yaml', 'w') as f:
        f.write(f'''path: /Dev/aah-ai/model/xb-yolo/datasets/{dataset_name} # dataset root dir
train: images/train # train images (relative to 'path') 998 images
val: images/train # val images (relative to 'path') 998 images
test: # test images (optional)

# Classes
names:
{"".join(classes)}''')
    exit(0)
    
    with tqdm(total=len(operators), desc='Total') as total_pbar:
        for i, operator in enumerate(operators):
            for facing in facing_list:
                res_dir = f'./exports/{facing}/{operator}'
                files = os.listdir(res_dir)
                files = [f for f in files if f.split('_')[0] in animation]
                for file in tqdm(files, desc=f'{operator}/{facing}', leave=False):
                    filename, _ = os.path.splitext(file)
                    imgs = overlay_img_on_background(f'./background.png', os.path.join(res_dir, file))
                    for idx, (rect, oper_img) in enumerate(imgs):
                        output_filename = f'{operator}_{facing}_{filename}_{idx}'
                        image_output_path = os.path.join(images_dir, f'{output_filename}.png')
                        labels_output_path = os.path.join(labels_dir, f'{output_filename}.txt')
                        
                        oper_img.save(image_output_path, format='PNG')
                        with open(labels_output_path, 'w') as f:
                            f.write(f'0 {rect}\n')
            total_pbar.update(1)
    
    