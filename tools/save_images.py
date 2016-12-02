# Imports
from skimage.color import label2rgb, rgb2gray, gray2rgb
from skimage import img_as_float
import numpy as np
import scipy.misc


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
             color_map, tag, void_label):
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in range(output.shape[0]):
        img = image_batch[j].transpose((1, 2, 0))
        img = norm_01(img, mask_batch[j], void_label)

        #img = image_batch[j].transpose((1, 2, 0))
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


# Save 4 images (Image, mask and result)
def save_img4(image_batch, mask_batch, output, output2, out_images_folder,
              epoch, color_map, tag, void_label):
    output[(mask_batch == void_label).nonzero()] = void_label
    images = []
    for j in range(output.shape[0]):
        img = image_batch[j].transpose((1, 2, 0))
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
        out_name = out_images_folder + tag + '_epoch' + str(epoch) + '_img' + str(j) + '.png'
        scipy.misc.toimage(combined_image).save(out_name)
        images.append(combined_image)
    return images


# Save 2 images (Image and mask)
def save_img2(img, mask, fname, color_map, void_label):
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
