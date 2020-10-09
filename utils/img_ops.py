import numpy as np
import tensorflow as tf


def center_crop(img_arr, crop_h, crop_w):
    h, w = img_arr.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))

    return img_arr[j:j + crop_h, i:i + crop_w, :]


# Pixel from [0,255] to [-1,1]
def bound_image_values(img_arr):
    return img_arr / 127.5 - 1


def bound_image_values_01(img_arr):
    return img_arr / 255.


def unbound_image_values(gen_img):
    img_0_1_bounded = (gen_img + 1.) / 2.
    img_0_255_bounded = (img_0_1_bounded * 255.).astype(np.uint8)
    return img_0_255_bounded


def unbound_images_values_01(gen_img):
    return (gen_img * 255.).astype(np.uint8)


def bound_0_to_1_tf(img_ph):
    assert img_ph.dtype == tf.uint8
    return tf.image.convert_image_dtype(img_ph, dtype=tf.float32, saturate=True)


def bound__1_to_1_tf(img_ph):
    assert img_ph.dtype == tf.uint8
    return tf.subtract(tf.div(tf.cast(img_ph, dtype=tf.float32), tf.constant(127.5, dtype=tf.float32)),
                       tf.constant(1., dtype=tf.float32))


def merge_images_to_grid(images, grid_size):
    h, w = images.shape[1], images.shape[2]
    big_img = np.zeros((h * grid_size[0], w * grid_size[1], 3), dtype=np.uint8)
    for idx, image in enumerate(images):  # idx=0,1,2,...,63
        i = idx % grid_size[1]  # column number
        j = idx // grid_size[1]  # row number
        big_img[j * h:j * h + h, i * w:i * w + w, :] = image
    return big_img


def compose_img_list_to_grid(images_list, transformer):
    assert isinstance(images_list, list)
    h, w = images_list[0].shape[1:3]
    imgs_per_row = images_list[0].shape[0]
    rows_num = len(images_list)
    padding = 3

    final_img = np.zeros((rows_num * h + padding*(rows_num-1), imgs_per_row * w + padding*(imgs_per_row-1), 3),
                         dtype=np.uint8)
    final_img.fill(255)
    for row_idx, images_row in enumerate(images_list):
        row_images_batch = images_list[row_idx]
        for col_idx in range(imgs_per_row):
            curr_img = row_images_batch[col_idx, :, :, :]
            curr_img = transformer(curr_img)

            i_pos, j_pos = row_idx * h + row_idx * padding, col_idx * w + col_idx * padding
            final_img[i_pos: i_pos + h, j_pos: j_pos + w, :] = curr_img

    return final_img


def save_gen_images(images, out_path, transformer, write_order='bgr'):
    import cv2

    grid_size = int(np.sqrt(images.shape[0]))
    reorganized_imgs = []
    for i in range(grid_size):
        reorganized_imgs.append(images[i*grid_size: grid_size*(i + 1)])

    big_img = compose_img_list_to_grid(reorganized_imgs, transformer)
    if write_order == 'rgb':
        big_img = big_img[:, :, ::-1]
    elif write_order != 'bgr':
        raise ValueError("Unknown write order: %s" % write_order)

    cv2.imwrite(out_path, big_img)
