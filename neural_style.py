import re
import os
import math
import scipy.misc
import numpy as np
from PIL import Image
from stylize import stylize
from collections import OrderedDict

'''
运行500-2000次会产生不错的结果。
--content-weight
--style-weight
--learning-rate
'''
content = "style/1.jpeg"  # 内容图片
styles = ["style/style1.jpg"]  # 一个或多个样式图像
output = "./result/result.jpg"
# default arguments
content_weight = 5e0  # 含量重量(默认5e0)
style_weight = 5e2  # 样式权重(默认5e2)
learning_rate = 1e1  # 学习率(默认1e1)
content_weight_blend = 1  # 指定内容传输层的系数，数值越小越抽象，该值应在[0.0, 1.0]。(默认值1)
tv_weight = 1e2  # 总变化正则化权重(默认1e2)
style_layer_weight_exp = 1  # 命令行参数可用于调整样式传输的“抽象”程度。较低的值意味着较精细的特征的样式传递将优于较粗糙的特征的样式传递，反之亦然。该值应在[0.0, 1.0]。(默认值1)
beta1 = 0.9  # Adam：beta1参数(默认0.9)
beta2 = 0.999  # Adam：beta2参数(默认0.999)
epsilon = 1e-08  # Adam：epsilon参数(默认1e-08)
STYLE_SCALE = 1.0  # (默认1.0)
iterations = 1500  # 迭代(默认1000)
network = 'imagenet-vgg-verydeep-19.mat'  # 网络参数的路径(默认为imagenet-vgg-verydeep-19.mat)
pooling = 'max'  # 最大池倾向于具有更好的细节样式传输，但是在低频细节级别上可能会有麻烦。(默认max 可选avg)
progress_write = False  # 将迭代进度数据写入OUTPUT目录 (默认Fasle)
progress_plot = False  # 将迭代进度数据绘制到OUTPUT目录 (默认Fasle)
checkpoint_output = "./result/output_{:05}.jpg"  # 检查点输出格式, 该值示例: ['output_{:05}.jpg', 'output_%%05d.jpg']  (默认None)
checkpoint_iterations = 100  # 检查点频率  (默认None)
width = None  # 输出宽度  (默认None)
style_scales = None  # 一个或多个样式标尺  (默认None)
print_iterations = None  # 统计打印频率  (默认None)
preserve_colors = None  # 仅样式转移，保留颜色。(默认None 可选True)
overwrite = True  # 即使已经有该名称的文件也要写入文件  (默认None)


def fmt_imsave(fmt, iteration):
    if re.match(r'^.*\{.*\}.*$', fmt):
        return fmt.format(iteration)
    elif '%' in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))


def main():
    key = 'TF_CPP_MIN_LOG_LEVEL'
    if key not in os.environ:
        os.environ[key] = '2'

    if not os.path.isfile(network):
        print("Network %s does not exist. (Did you forget to download it?)" % network)

    if [checkpoint_iterations, checkpoint_output].count(None) == 1:
        print("use either both of checkpoint_output and checkpoint_iterations or neither")

    if checkpoint_output is not None:
        if re.match(r'^.*(\{.*\}|%.*).*$', checkpoint_output) is None:
            print("To save intermediate images, the checkpoint_output parameter must contain placeholders (e.g. `foo_{}.jpg` or `foo_%d.jpg`)")

    content_image = imread(content)
    style_images = [imread(style) for style in styles]

    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        style_scale = STYLE_SCALE
        if style_scales is not None:
            style_scale = style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                target_shape[1] / style_images[i].shape[1])

    style_blend_weights = None  # 样式混合权重
    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0/len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight/total_blend_weight for weight in style_blend_weights]

    initial = None  # 初始图像
    initial_noiseblend = None  # 初始图像与标准化噪声混合的比率(如果未指定初始图像，则使用内容图像)(默认None)
    if initial is not None:
        initial = scipy.misc.imresize(imread(initial), content_image.shape[:2])
        # Initial guess is specified, but not noiseblend - no noise should be blended
        if initial_noiseblend is None:
            initial_noiseblend = 0.0
    else:
        # Neither inital, nor noiseblend is provided, falling back to random
        # generated initial guess
        if initial_noiseblend is None:
            initial_noiseblend = 1.0
        if initial_noiseblend < 1.0:
            initial = content_image

    # try saving a dummy image to the output path to make sure that it's writable
    if os.path.isfile(output) and not overwrite:
        raise IOError("%s already exists, will not replace it without "
                      "the '--overwrite' flag" % output)
    try:
        imsave(output, np.zeros((500, 500, 3)))
    except:
        raise IOError('%s is not writable or does not have a valid file '
                      'extension for an image file' % output)

    loss_arrs = None
    for iteration, image, loss_vals in stylize(
        network=network,
        initial=initial,
        initial_noiseblend=initial_noiseblend,
        content=content_image,
        styles=style_images,
        preserve_colors=preserve_colors,
        iterations=iterations,
        content_weight=content_weight,
        content_weight_blend=content_weight_blend,
        style_weight=style_weight,
        style_layer_weight_exp=style_layer_weight_exp,
        style_blend_weights=style_blend_weights,
        tv_weight=tv_weight,
        learning_rate=learning_rate,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,
        pooling=pooling,
        print_iterations=print_iterations,
        checkpoint_iterations=checkpoint_iterations,
    ):
        if (image is not None) and (checkpoint_output is not None):
            imsave(fmt_imsave(checkpoint_output, iteration), image)
        if (loss_vals is not None) \
                and (progress_plot or progress_write):
            if loss_arrs is None:
                itr = []
                loss_arrs = OrderedDict((key, []) for key in loss_vals.keys())
            for key,val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    imsave(output, image)

    if progress_write:
        fn = "{}/progress.txt".format(os.path.dirname(output))
        tmp = np.empty((len(itr), len(loss_arrs)+1), dtype=float)
        tmp[:,0] = np.array(itr)
        for ii,val in enumerate(loss_arrs.values()):
            tmp[:,ii+1] = np.array(val)
        np.savetxt(fn, tmp, header=' '.join(['itr'] + list(loss_arrs.keys())))


    if progress_plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig,ax = plt.subplots()
        for key, val in loss_arrs.items():
            ax.semilogy(itr, val, label=key)
        ax.legend()
        ax.set_xlabel("iterations")
        ax.set_ylabel("loss")
        fig.savefig("{}/progress.png".format(os.path.dirname(output)))


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)

if __name__ == '__main__':
    main()
