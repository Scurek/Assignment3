import math
import os
import cv2
from PIL import Image, ImageDraw


def saveImg(img, prediction_list, output_path, folder, img_name, color=(255, 0, 0), thickness=2):
    # print(prediction_list)
    img = img.copy()
    for (x, y, w, h) in prediction_list:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), thickness)
    # ax = plt.subplot(111)
    # ax.imshow(img)
    # plt.savefig(path + "output_" + base_name)
    full_path = os.path.join(output_path, folder)
    if not os.path.isdir(full_path):
        os.mkdir(full_path)
    full_path = os.path.join(full_path, img_name) + ".png"
    cv2.imwrite(full_path, img)
    return full_path


def saveAll(img_list, output_path, folder, img_name, labels, color_list, width=480, height=360, max_rows=100,
            max_columns=10):
    columns = min(int(math.ceil(math.sqrt(len(img_list)))), max_columns)
    rows = min(int(math.ceil(len(img_list) / columns)), max_rows)
    base = Image.new("RGBA", (columns * width, rows * height))
    draw = ImageDraw.Draw(base)

    for i in range(len(img_list)):
        img = Image.open(img_list[i])
        column = i % columns
        row = int(i / columns)
        x = column * width
        y = row * height
        base.paste(img, (x, y))
        draw.text((x + 5, y + height - 20), labels[i], align="left", fill=color_list[i])

    full_path = os.path.join(output_path, folder)
    if not os.path.isdir(full_path):
        os.mkdir(full_path)
    full_path = os.path.join(full_path, img_name) + ".png"
    base.save(full_path)
    return full_path
