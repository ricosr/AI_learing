# -*- encoding:utf-8 -*-

import numpy as np
import scipy.misc
import reload_vgg16
import tensorflow as tf
from functools import reduce
from PIL import Image
from sys import stderr


class CreateImg:
    def __init__(self, content_img, style_img, vgg16_path, output_img):
        self.content_img = content_img
        self.style_img = style_img
        self.output_img = output_img
        self.load_vgg16(vgg16_path)

    def extract_content(self):
        content_img = self.load_img(self.content_img)
        self.content_shape = (1,) + content_img.shape   # ????????
        vgg16 = reload_vgg16.Vgg16(self.vgg16_dict, "content", self.content_shape)
        content_img = np.reshape(content_img, self.content_shape)
        self.content_conv3_1 = vgg16.train("content", content_img)

    def extract_style(self):
        style_img = self.load_img(self.style_img)
        style_shapes = (1,) + style_img.shape
        vgg16 = reload_vgg16.Vgg16(self.vgg16_dict, "style", style_shapes)
        style_img = np.reshape(style_img, style_shapes)
        style_layers_tmp = vgg16.train("style", style_img)
        style_layers_tmp = map((lambda each_layer_style: np.reshape(each_layer_style, (-1, each_layer_style.shape[3]))), style_layers_tmp)
        style_layers_tmp = map((lambda each_layer_style: np.matmul(each_layer_style.T, each_layer_style) / each_layer_style.size), style_layers_tmp)
        self.style_keys = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
        self.style_layers = dict(zip(self.style_keys, style_layers_tmp))

    def init_new_img(self):
        # if initial is None:
            # noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        initial = tf.random_normal(self.content_shape) * 0.256  # 初始化一个图片数据
        # else:
        #     initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
        #     initial = initial.astype('float32')
        #     noise = np.random.normal(size=shape, scale=np.std(content) * 0.1)
        #     initial = (initial) * initial_content_noise_coeff + (tf.random_normal(shape) * 0.256) * (
        #     1.0 - initial_content_noise_coeff)
        self.new_img = tf.Variable(initial)  # 包装初始化一个图片数据
        self.vgg16_new = reload_vgg16.Vgg16(vgg16_npy=self.vgg16_dict, step="optimize", new_img=self.new_img)
        # net = vgg.net_preloaded(vgg_weights, image, pooling)  # 将初始化图片扔到vgg里面

    def optimize_diff(self):
        # print(self.vgg16_new.conv3_1.get_shape())
        # print(self.content_conv3_1.shape)
        content_loss = 2 * (tf.nn.l2_loss(self.vgg16_new.conv3_1 - self.content_conv3_1) / self.content_conv3_1.size)
        # print(content_loss.eval())
        style_losses = []
        # print(self.vgg16_new.layers_ls)
        for i in range(len(self.style_keys)):
            layer = self.vgg16_new.layers_ls[i]
            _, height, width, number = map(lambda i: i.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = self.style_layers[self.style_keys[i]]
            # print(gram.get_shape())
            # print(style_gram.shape)
            style_losses.append(0.2 * 2 * (tf.nn.l2_loss(gram - style_gram) / style_gram.size))
        style_loss = reduce(tf.add, style_losses)
        loss = 5e2 * content_loss + 5e0 * style_loss
        train_step = tf.train.AdamOptimizer(1e1, 0.9, 0.999, 1e-08).minimize(loss)    # ??????????

        best_loss = float('inf')  # 正无穷
        best = None
        # TODO
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            stderr.write('Optimization started...\n')
            times = 500
            for i in range(times):
                print("finish {0}%\r".format(int(i/times*100)), end='')
                train_step.run()
                last_step = (i == times - 1)
                if last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = self.new_img.eval()
                    img_out = best.reshape(self.content_shape[1:])
                    print("finish 100%\r")
                    return img_out

    def load_img(self, path):
        img = scipy.misc.imread(path).astype(np.float)
        # if len(img.shape) == 2:
        #     # gray imgture
        #     img = np.dstack((img, img, img))
        # elif img.shape[2] == 4:
        #     # PNG with alpha channel
        #     img = img[:, :, :3]
        return img

    def save_img(self, out_path, output_img):
        img = np.clip(output_img, 0, 255).astype(np.uint8)
        # img = np.clip(self.new_img, 0, 255).astype(np.uint8)
        # Image.fromarray(img).save(self.output_img, quality=95)
        Image.fromarray(img).save(out_path, quality=95)

    def load_vgg16(self, vgg16_path):
        try:
            self.vgg16_dict = np.load(vgg16_path, encoding='latin1').item()
        except FileNotFoundError:
            print('Please download VGG16 parameters at here https://mega.nz/#!YU1FWJrA!O1ywiCS2IiOlUCtCpI6HTJOMrneN-Qdv3ywQP5poecM')

def main():
    output_img_path = "upload2.jpg"
    create_img = CreateImg(content_img="upload.jpg", style_img="examples/1-style.jpg", vgg16_path="vgg16.npy", output_img=output_img_path)
    create_img.save_img(output_img_path, np.zeros((500, 500, 3)))
    create_img.extract_content()
    create_img.extract_style()
    create_img.init_new_img()
    new_img = create_img.optimize_diff()
    create_img.save_img(create_img.output_img, new_img)


if __name__ == '__main__':
    main()
