import tensorflow as tf
from tensorflow.python.ops import gen_image_ops

from constants import COEF_8, COEF_16


def touint8_tf(img_t, name=None):
    img_t = tf.clip_by_value(img_t, clip_value_min=0., clip_value_max=1.) * COEF_8
    return tf.cast(img_t, dtype=tf.uint8, name=name)


def touint16_tf(img_t, name=None):
    img_t = tf.clip_by_value(img_t, clip_value_min=0., clip_value_max=1.) * COEF_16
    return tf.cast(img_t, dtype=tf.uint16, name=name)


# ==================== Read/Write image utils. ==================== #


def read_img_tf(read_path,
                dtype=tf.uint16,
                name="input_img",
                expand_dims=False):
    # Image reading.
    x = tf.io.decode_png(tf.io.read_file(read_path), dtype=dtype, name=name)  # This op also supports decoding JPEGs and non-animated GIFs

    # Image processing.
    if dtype == tf.uint8:
        x = tf.cast(x, dtype=tf.float32, name=name+"__float") / COEF_8
    elif dtype == tf.uint16:
        x = tf.cast(x, dtype=tf.float32, name=name+"__float") / COEF_16
    else:
        raise Exception("src/imgtf_utils.py: "
                        "def read_img_tf(...): "
                        "error: `dtype` should be either tf.uint8 or tf.uint16, but found {}".format(dtype))
    if expand_dims:
        x = tf.expand_dims(x, 0)

    return tf.cast(x, dtype=tf.float32, name=name+"_float")


# ==================== Image transforms utils. ==================== #

SCALE_METHODS = {
    "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
    "bilin": tf.image.ResizeMethod.BILINEAR,
    "bicub": tf.image.ResizeMethod.BICUBIC,
}


def upscale_tf(img_t, factor, method):
    """
    Method for upscaling images with given ``factor`` using ``method``.
    Args:
        img_t: Input images to transform.
        factor: Image scaling factor.
        method: Method for interpolating while scaling.

    Returns:
        Scaled images.
    """
    assert method in SCALE_METHODS

    b, h, w, c = img_t.get_shape()
    size = [tf.cast(h * factor, tf.int32), tf.cast(w * factor, tf.int32)]

    if method == "nearest":
        out_t = gen_image_ops.resize_nearest_neighbor(img_t,
                                                      size,
                                                      half_pixel_centers=False,
                                                      align_corners=False)
    elif method == "bilin":
        out_t = gen_image_ops.resize_bilinear(img_t,
                                              size,
                                              half_pixel_centers=False,
                                              align_corners=False)
    elif method == "bicub":
        out_t = gen_image_ops.resize_bicubic(img_t,
                                             size,
                                             half_pixel_centers=False,
                                             align_corners=False)
    return out_t


def downscale_tf(img_t, factor, method):
    """
    Method for downscaling images with given ``factor`` using ``method``.
    Args:
        img_t: Input images to transform.
        factor: Image scaling factor.
        method: Method for interpolating while scaling.

    Returns:
        Scaled images.
    """
    assert method in SCALE_METHODS

    b, h, w, c = img_t.get_shape()
    size = [tf.cast(h // factor, tf.int32), tf.cast(w // factor, tf.int32)]

    if method == "nearest":
        out_t = gen_image_ops.resize_nearest_neighbor(img_t,
                                                      size,
                                                      half_pixel_centers=False,
                                                      align_corners=False)
    elif method == "bilin":
        out_t = gen_image_ops.resize_bilinear(img_t,
                                              size,
                                              half_pixel_centers=False,
                                              align_corners=False)
    elif method == "bicub":
        out_t = gen_image_ops.resize_bicubic(img_t,
                                             size,
                                             half_pixel_centers=False,
                                             align_corners=False)
    return out_t


# ==================== Other utils. ==================== #


def get_dummy_tensor(batch_size=1, height=512, width=512, channels=1, dtype=tf.float32, data_format="NHWC"):
    if data_format == "NHWC":
        return tf.random.normal(shape=(batch_size, height, width, channels), dtype=dtype)
    else:
        return tf.random.normal(shape=(batch_size, channels, height, width), dtype=dtype)
