from typing import Callable
import numpy as np
import tensorflow as tf


def init_images(content_im: np.ndarray, style_im: np.ndarray) -> tuple[tf.Tensor]:
    """ returns content_im, style_im, and generated_im as tf.Tensors """
    content_im = tf.constant(content_im, dtype=tf.float32)[tf.newaxis, :]
    style_im = tf.constant(style_im, dtype=tf.float32)[tf.newaxis, :]
    im = tf.Variable(content_im, trainable=True)
    return content_im, style_im, im


def clip_0_1(image: tf.Tensor):
    return tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)


def high_pass_x_y(image: tf.Tensor):
    x_var = image[:, :, 1:, :] - image[:, :, :-1, :]
    y_var = image[:, 1:, :, :] - image[:, :-1, :, :]
    return x_var, y_var


def total_variation_loss(image: tf.Tensor):
    x_deltas, y_deltas = high_pass_x_y(image)
    return tf.reduce_sum(tf.abs(x_deltas)) + tf.reduce_sum(tf.abs(y_deltas))


def gram_matrix(t: tf.Tensor) -> tf.Tensor:
    result = tf.linalg.einsum("bijc,bijd->bcd", t, t)
    sh = tf.shape(t)
    num_locations = sh[1] * sh[2]
    return result / tf.cast(num_locations, tf.float32)


def print_vgg_layer_names():
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    for layer in vgg.layers:
        print(layer.name)


def vgg_layers(layer_names: list[str]) -> tf.keras.Model:
    """ Creates a vgg model that returns a list of intermediate output layers """
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    return tf.keras.Model([vgg.input], outputs)


def avg_mse(outputs: dict[str, tf.Tensor], targets: dict[str, tf.Tensor]) -> tf.Tensor:
    """ calculates mse for each layer, then returns mean across all layers """
    layer_losses = [
        tf.reduce_mean((outputs[name] - targets[name]) ** 2) for name in outputs.keys()
    ]
    return tf.reduce_mean(layer_losses)


class StyleContentLoss:
    def __init__(self, content_targets, style_targets, style_weight, content_weight):
        self.content_targets = content_targets
        self.style_targets = style_targets

        self.style_weight = style_weight
        self.content_weight = content_weight

    def __call__(self, outputs: dict[str, dict]):
        style_loss = avg_mse(outputs["style"], self.style_targets)
        content_loss = avg_mse(outputs["content"], self.content_targets)
        return style_loss * self.style_weight + content_loss * self.content_weight


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg_trainable = False

    def call(self, t: tf.Tensor) -> dict[str, dict]:
        t = t * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(t)

        outputs = self.vgg(preprocessed_input)
        style_outputs = outputs[: self.num_style_layers]
        content_outputs = outputs[self.num_style_layers :]

        style_outputs = [gram_matrix(out) for out in style_outputs]

        content_dict = {n: val for n, val in zip(self.content_layers, content_outputs)}
        style_dict = {n: val for n, val in zip(self.style_layers, style_outputs)}
        return {"content": content_dict, "style": style_dict}


def get_train_step(
    model: tf.keras.Model,
    optim: tf.keras.optimizers.Optimizer,
    styleContentLoss: StyleContentLoss,
) -> Callable:
    @tf.function()
    def train_step(image: tf.Tensor):
        with tf.GradientTape() as tape:
            outputs = model(image)
            loss = styleContentLoss(outputs)
            loss += total_variation_loss(image)

        grad = tape.gradient(loss, image)
        optim.apply_gradients([(grad, image)])
        image.assign(clip_0_1(image))

    return train_step