from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fontsize = 50
font = ImageFont.truetype('./NotoSansCJKjp-Regular.otf', size=fontsize, encoding='utf-8')
df_train = pd.read_csv('../input/train.csv')
unicode_map = {codepoint: char for codepoint, char in pd.read_csv('../input/unicode_translation.csv').values}


def visualize(image_fn, labels):
    labels = np.array(labels.split(' ')).reshape(-1, 5)
    imsource = Image.open(image_fn).convert('RGBA')
    bbox_canvas = Image.new('RGBA', imsource.size)
    char_canvas = Image.new('RGBA', imsource.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    char_draw = ImageDraw.Draw(char_canvas)

    for codepoint, x, y, w, h in labels:
        x, y, w, h = int(x), int(y), int(w), int(h)
        char = unicode_map[codepoint]
        bbox_draw.rectangle((x, y, x + w, y + h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255))
        char_draw.text((x + w + fontsize / 4, y + h / 2 - fontsize), char, fill=(0, 0, 255, 255), font=font)

    imsource = Image.alpha_composite(Image.alpha_composite(imsource, bbox_canvas), char_canvas)
    imsource = imsource.convert('RGB')
    return np.asarray(imsource)


for i in range(10):
    img, labels = df_train.values[np.random.randint(len(df_train))]
    viz = visualize('../input/train_images/{}.jpg'.format(img), labels)

    plt.figure(figsize=(15, 15))
    plt.title(img)
    plt.imshow(viz, interpolation='lanczos')
    plt.show()
