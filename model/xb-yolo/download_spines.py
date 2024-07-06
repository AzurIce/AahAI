import requests
import os
from tqdm import tqdm

def download_file(url, local_filepath):
    if os.path.exists(local_filepath):
        print(f'{local_filepath} already exists, skipping...')
        return
    
    with requests.get(url, stream=True) as response:
        response.raise_for_status()
        with open(local_filepath, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

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

if __name__ == '__main__':
    total_iterations = len(operators)  # operators * facing * ext
    with tqdm(total=total_iterations, desc="Total") as pbar_total:
        for operator in operators:
            with tqdm(total=2 * 3, desc=f"{operator}", leave=False) as pbar:
                for facing in ['front', 'back']:
                    for ext in ['skel', 'atlas', 'png']:
                        filename = f'{operator}.{ext}'
                        output_dir = f'./spines/{facing}/{operator}'
                        os.makedirs(output_dir, exist_ok=True)
                        
                        url = f'https://torappu.prts.wiki/assets/char_spine/{operator}/defaultskin/{facing}/{filename}'
                        path = os.path.join(output_dir, filename)
                        download_file(url, path)
                        pbar.update(1)
            pbar_total.update(1)
    