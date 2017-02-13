# Imports
from skimage.color import rgb2gray, gray2rgb
from skimage import img_as_float
import numpy as np
import scipy.misc
import os
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import math
import skimage.io as io
from keras import backend as K
dim_ordering = K.image_dim_ordering()


# Normalizes image to 0-1 range
def norm_01(img, y, void_label):
    # Normalize image
    max_v = np.max(img)
    min_v = np.min(img)
    img = (img-min_v)/(max_v-min_v)

    # Compute the void mask
    y = y.reshape((y.shape[0], y.shape[1], 1))
    mask = np.ones_like(y).astype('int32')
    mask[y == void_label] = 0.
    mask = np.repeat(mask, 3, axis=2)

    # Set void values to 0
    img = img*mask

    return img


# Finds the best font size
def find_font_size(max_width, classes, font_file, max_font_size=100):

    draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))

    # Find the maximum font size that all labels fit into the box width
    n_classes = len(classes)
    for c in range(n_classes):
        text = classes[c]
        for s in range(max_font_size, 1, -1):
            font = ImageFont.truetype(font_file, s)
            txt_size = draw.textsize(text, font=font)
            # print('c:{} s:{} txt_size:{}'.format(c, s, txt_size))
            if txt_size[0] <= max_width:
                max_font_size = s
                break

    # Find the maximum box height needed to fit the labels
    max_font_height = 1
    font = ImageFont.truetype(font_file, max_font_size)
    for c in range(n_classes):
        max_font_height = max(max_font_height,
                              draw.textsize(text, font=font)[1])

    return max_font_size, int(max_font_height*1.25)


# Draw class legend in an image
def draw_legend(w, color_map, classes, n_lines=3, txt_color=(255, 255, 255),
                font_file="fonts/Cicle_Gordita.ttf"):

    # Compute legend sizes
    n_classes = len(color_map)
    n_classes_per_line = int(math.ceil(float(n_classes) / n_lines))
    class_width = w/n_classes_per_line
    font_size, class_height = find_font_size(class_width, classes, font_file)
    font = ImageFont.truetype(font_file, font_size)

    # Create PIL image
    img_pil = Image.new('RGB', (w, n_lines*class_height))
    draw = ImageDraw.Draw(img_pil)

    # Draw legend
    for i in range(n_classes):
        # Get color and label
        color = color_map[i]
        text = classes[i]

        # Compute current row and col
        row = i/n_classes_per_line
        col = i % n_classes_per_line

        # Draw box
        box_pos = [class_width*col, class_height*row,
                   class_width*(col+1), class_height*(row+1)]
        draw.rectangle(box_pos, fill=color, outline=None)

        # Draw text
        txt_size = draw.textsize(text, font=font)[0]
        txt_pos = [box_pos[0]+((box_pos[2]-box_pos[0])-txt_size)/2, box_pos[1]]
        draw.text(txt_pos, text, txt_color, font=font)

    return np.asarray(img_pil)


# Converts a label mask to RGB to be shown
def my_label2rgb(labels, colors, bglabel=None, bg_color=(0., 0., 0.)):
    output = np.zeros(labels.shape + (3,), dtype=np.float64)
    for i in range(len(colors)):
        if i != bglabel:
            output[(labels == i).nonzero()] = colors[i]
    if bglabel is not None:
        output[(labels == bglabel).nonzero()] = bg_color
    return output


# Converts a label mask to RGB to be shown and overlaps over an image
def my_label2rgboverlay(labels, colors, image, bglabel=None,
                        bg_color=(0., 0., 0.), alpha=0.2):
    image_float = gray2rgb(img_as_float(rgb2gray(image)))
    label_image = my_label2rgb(labels, colors, bglabel=bglabel,
                               bg_color=bg_color)
    output = image_float * alpha + label_image * (1 - alpha)
    return output


# Save 3 images (Image, mask and result)
def save_img3(image_batch, mask_batch, output, out_images_folder, epoch,
             color_map, classes, tag, void_label, n_legend_rows=1):
    # print('output shape: ' + str(output.shape))
    # print('Mask shape: ' + str(mask_batch.shape))
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in range(output.shape[0]):
        img = image_batch[j]
        if dim_ordering == 'th':
            img = img.transpose((1, 2, 0))

        #img = norm_01(img, mask_batch[j], void_label)*255
        img = norm_01(img, mask_batch[j], -1)*255

        #img = image_batch[j].transpose((1, 2, 0))
        label_out = my_label2rgb(output[j], bglabel=void_label,
                                 colors=color_map)
        label_mask = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                         image=img, bglabel=void_label,
                                         alpha=0.3)
        label_overlay = my_label2rgboverlay(output[j], colors=color_map,
                                            image=img, bglabel=void_label,
                                            alpha=0.3)

        combined_image = np.concatenate((img, label_mask, label_out,
                                         label_overlay), axis=1)

        legend = draw_legend(combined_image.shape[1], color_map, classes,
                             n_lines=n_legend_rows)
        combined_image = np.concatenate((combined_image, legend))

        out_name = os.path.join(out_images_folder, tag + '_epoch' + str(epoch) + '_img' + str(j) + '.png')
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


# Save 4 images (Image, mask and result)
def save_img4(image_batch, mask_batch, output, output2, out_images_folder,
              epoch, color_map, tag, void_label):
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in range(output.shape[0]):
        img = image_batch[j]
        if dim_ordering == 'th':
            img = img.transpose((1, 2, 0))
        img = norm_01(img, mask_batch[j], void_label)

        label_out = my_label2rgb(output[j], bglabel=void_label,
                                 colors=color_map)

        label_out2 = my_label2rgb(output2[j], bglabel=void_label,
                                  colors=color_map)

        label_mask = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                         image=img, bglabel=void_label,
                                         alpha=0.2)
        label_overlay = my_label2rgboverlay(output[j], colors=color_map,
                                            image=img, bglabel=void_label,
                                            alpha=0.5)

        combined_image = np.concatenate((img, label_mask, label_out, label_out2,
                                         label_overlay), axis=1)
        out_name = os.path.join(out_images_folder, tag + '_epoch' + str(epoch) + '_img' + str(j) + '.png')
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


# Save 2 images (Image and mask)
def save_img2(img, mask, fname, color_map, void_label):
    if dim_ordering == 'th':
        img = img.transpose((1, 2, 0))

    mask = mask.reshape(mask.shape[1:3])
    img = norm_01(img, mask, void_label)

    label_mask = my_label2rgboverlay(mask,
                                     colors=color_map,
                                     image=img,
                                     bglabel=void_label,
                                     alpha=0.2)
    combined_image = np.concatenate((img, label_mask),
                                    axis=1)
    scipy.misc.toimage(combined_image).save(fname)
