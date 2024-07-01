import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg


def draw_boxes(fn_image_file, fn_txt_file):
    # Read image
    img = mpimg.imread(fn_image_file)

    # Plot image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_aspect('equal')

    # Read and process bounding box data from txt_file
    with open(fn_txt_file, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        name = parts[0]
        center_x = float(parts[1])
        center_y = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        # Convert center and size to absolute coordinates
        left = center_x - width / 2
        bottom = center_y - height / 2

        # Get image dimensions
        img_height, img_width, _ = img.shape

        # Scale coordinates to match image size
        left *= img_width
        bottom *= img_height
        width *= img_width
        height *= img_height

        # Create rectangle patch
        rect = patches.Rectangle((left, bottom), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(left, bottom-5, name, fontsize=8, color='r', verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    ax.set_xlim(0, img_width)
    ax.set_ylim(img_height, 0)  # Invert y-axis to match image coordinates

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bounding Boxes')

    plt.show()


# Example usage:
image_file = 'test/image/0_Lancet_3_1_0.png'
txt_file = 'test/label/0_Lancet_3_1_0.txt'
draw_boxes(image_file, txt_file)



