import io
import re
import os
import math
import scipy.misc
import numpy as np
from PIL import Image
from stylize import stylize
from collections import OrderedDict
from fastapi import FastAPI, File, UploadFile, Body

app = FastAPI()

@app.post("/uploadfile/")
async def style_fusion(content_image: UploadFile = File(...),
                       style_img: UploadFile = File(...),
                       content_weight: int = Body(5e0, description="(默认5e0), 内容图片权重,值越大内容图片细节保存越多. "),
                       style_weight: int = Body(5e2, description="(默认5e2), 样式图片权重, 值越大样式内容细节保存越多."),
                       learning_rate: int = Body(1e1, description="(默认1e1), 学习率, 学习图像风格的快慢."),
                       content_weight_blend: int = Body(1, description="(默认值1), 指定内容传输层的系数,数值越小越抽象,该值应在[0.0, 1.0]", ),
                       tv_weight: int = Body(1e2, description="(默认1e2), 总变化正则化权重"),
                       style_layer_weight_exp: int = Body(1, description="(默认值1), 命令行参数可用于调整样式传输的“抽象”程度。较低的值意味着较精细的特征的样式传递将优于较粗糙的特征的样式传递,反之亦然。该值应在[0.0, 1.0]。"),
                       beta1: float = Body(0.9, description="(默认0.9), Adam：beta1参数"),
                       beta2: float = Body(0.999, description="(默认0.999), Adam：beta2参数"),
                       epsilon: float = Body(1e-08, description="(默认1e-08), Adam：epsilon参数"),
                       style_scale: float = Body(1.0, description="(默认1.0)"),
                       iterations: int = Body(1000, description="(默认1000), 迭代次数, 用于计算风格图和内容图像相似的次数"),
                       pooling: str = Body('max', description="(默认max 可选avg), 最大池化倾向于具有更好的细节样式传输, 但是在低频细节级别上可能会有麻烦"),
                       progress_write: bool = Body(False, description="(默认Fasle), 将迭代进度数据写入OUTPUT目录"),
                       progress_plot: bool = Body(False, description="(默认Fasle), 将迭代进度数据绘制到OUTPUT目录"),
                       checkpoint_iterations: int = Body(100, description="(默认100), 可选None, 和其它整数型频率"),
                       width: int = Body(None, description="(默认None), 输出图像宽度"),
                       style_scales: int = Body(None, description="(默认None), 一个或多个样式标尺"),
                       print_iterations: bool = Body(None, description="(默认None), 统计打印频率"),
                       preserve_colors: bool = Body(None, description="(默认None 可选True), 仅样式转移, 保留颜色"),
                       overwrite: bool = Body(True, description="(默认True, 可选None), 覆盖已保存的文件"),
                       style_blend_weights: float = Body(None, description="(默认None 一般0.2), 样式混合权重, 该值应在[0.0, 1.0]"),
                       initial: bool = Body(None, description="(默认None), 初始图像"),
                       initial_noiseblend: float = Body(None, description="(默认None), 初始图像与标准化噪声混合的比率(如果未指定初始图像,则使用内容图像)")
                       ):
    content_image = await content_image.read()
    style_img = await style_img.read()
    content_image = io.BytesIO(content_image)
    style_img = io.BytesIO(style_img)
    content_image = imread(content_image)
    style_images = [imread(style_img)]

    dir = "./result"
    if not os.path.exists(dir):
        os.makedirs(dir)
    network = './imagenet-vgg-verydeep-19.mat'  # 网络参数的路径(默认为imagenet-vgg-verydeep-19.mat)
    output = "./result/result.jpg"  # 默认融合风格后结果图像的输出路径.
    checkpoint_output = "./result/output_{:05}.jpg"  # (默认'./result/output_{:05}.jpg)', 可选 None, 用于保存每个阶段的风格迁移图像.
    key = 'TF_CPP_MIN_LOG_LEVEL'
    if key not in os.environ:
        os.environ[key] = '2'
    if not os.path.isfile(network):
        print("Network %s does not exist. (Did you forget to download it?)" % network)

    if [checkpoint_iterations, checkpoint_output].count(None) == 1:
        print("use either both of checkpoint_output and checkpoint_iterations or neither")

    if checkpoint_output is not None:
        if re.match(r'^.*(\{.*\}|%.*).*$', checkpoint_output) is None:
            print(
                "To save intermediate images, the checkpoint_output parameter must contain placeholders (e.g. `foo_{}.jpg` or `foo_%d.jpg`)")

    # content_image = imread(content)
    # style_images = [imread(style) for style in styles]

    if width is not None:
        new_shape = (int(math.floor(float(content_image.shape[0]) /
                                    content_image.shape[1] * width)), width)
        content_image = scipy.misc.imresize(content_image, new_shape)
    target_shape = content_image.shape
    for i in range(len(style_images)):
        if style_scales is not None:
            style_scale = style_scales[i]
        style_images[i] = scipy.misc.imresize(style_images[i], style_scale *
                                              target_shape[1] / style_images[i].shape[1])

    if style_blend_weights is None:
        # default is equal weights
        style_blend_weights = [1.0 / len(style_images) for _ in style_images]
    else:
        total_blend_weight = sum(style_blend_weights)
        style_blend_weights = [weight / total_blend_weight for weight in style_blend_weights]

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
            network=str(network),
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
            for key, val in loss_vals.items():
                loss_arrs[key].append(val)
            itr.append(iteration)

    imsave(output, image)

    if progress_write:
        fn = "{}/progress.txt".format(os.path.dirname(output))
        tmp = np.empty((len(itr), len(loss_arrs) + 1), dtype=float)
        tmp[:, 0] = np.array(itr)
        for ii, val in enumerate(loss_arrs.values()):
            tmp[:, ii + 1] = np.array(val)
        np.savetxt(fn, tmp, header=' '.join(['itr'] + list(loss_arrs.keys())))

    if progress_plot:
        import matplotlib
        matplotlib.use('Agg')
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots()
        for key, val in loss_arrs.items():
            ax.semilogy(itr, val, label=key)
        ax.legend()
        ax.set_xlabel("iterations")
        ax.set_ylabel("loss")
        fig.savefig("{}/progress.png".format(os.path.dirname(output)))
    return "save img success..."


def fmt_imsave(fmt, iteration):
    if re.match(r'^.*\{.*\}.*$', fmt):
        return fmt.format(iteration)
    elif '%' in fmt:
        return fmt % iteration
    else:
        raise ValueError("illegal format string '{}'".format(fmt))


def imread(path):
    img = scipy.misc.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img, img, img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:, :, :3]
    return img


def imsave(path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path, quality=95)
