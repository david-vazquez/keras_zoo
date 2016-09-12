# Imports
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage import img_as_float
import numpy as np
import scipy.misc


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


# Save images
def save_img(image_batch, mask_batch, output, out_images_folder, epoch,
             color_map, tag, void_label):
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in xrange(output.shape[0]):
        img = image_batch[j].transpose((1, 2, 0)) / 255.
        label_out = my_label2rgb(output[j], bglabel=void_label,
                                 colors=color_map)
        label_mask = my_label2rgboverlay(mask_batch[j], colors=color_map,
                                         image=img, bglabel=void_label,
                                         alpha=0.2)
        label_overlay = my_label2rgboverlay(output[j], colors=color_map,
                                            image=img, bglabel=void_label,
                                            alpha=0.5)

        combined_image = np.concatenate((img, label_mask, label_out,
                                         label_overlay), axis=1)
        out_name = out_images_folder + tag + '_epoch' + str(epoch) + '_img' + str(j) + '.png'
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


def save_img2(img, mask, fname, color_map, void_label):
    img = img.transpose((1, 2, 0)) / 255.
    mask = mask.reshape(mask.shape[1:3])
    label_mask = my_label2rgboverlay(mask,
                                     colors=color_map,
                                     image=img,
                                     bglabel=void_label,
                                     alpha=0.2)
    combined_image = np.concatenate((img, label_mask),
                                    axis=1)
    scipy.misc.toimage(combined_image).save(fname)
