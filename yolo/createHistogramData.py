import os
from PIL import Image
from tqdm import tqdm

image_folder_path = "extraDatasets/image"
label_folder_path = "extraDatasets/label"
image_result_path = "HistogramData/image"
label_result_path = "HistogramData/label"

for filename in tqdm(os.listdir(image_folder_path)):
    if filename.endswith('.png'):
        base_filename, extension = os.path.splitext(filename)

        full_image_path = os.path.join(image_folder_path, filename)
        full_label_path = os.path.join(label_folder_path, base_filename + ".txt")

        image = Image.open(full_image_path)
        # Read label file
        with open(full_label_path, 'r') as label_file:
            label_content = label_file.read()

        for i in range(60, 81, 5):
            dark_factor = i / 100.0
            dark_image = Image.eval(image, lambda x: x * dark_factor)
            dark_image_filename = f"{base_filename}_dark_{i}.png"
            dark_image_path = os.path.join(image_result_path, dark_image_filename)
            dark_image.save(dark_image_path)
            # Save label content to a new file
            new_label_file_path = os.path.join(label_result_path, f"{base_filename}_dark_{i}.txt")
            with open(new_label_file_path, 'w') as new_label_file:
                new_label_file.write(label_content)

