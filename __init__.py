import csv
import json
import os

import numpy as np
import pylab
from PIL import Image
from flask import Flask, render_template, send_from_directory, request, Response

from flask_cors import *

import cv2 as cv

from matplotlib import pyplot as plt
from skimage import measure, filters, img_as_ubyte
from skimage.feature import canny

from strUtil import pic_str

from fcm import get_centroids, get_label, get_init_fuzzy_mat, fcm
from lv_set.find_lsf import find_lsf
from lv_set.drlse import get_params

from preprocess import gamma_trans,clahe_trans

# 配置Flask路由，使得前端可以访问服务器中的静态资源
app = Flask(__name__, static_folder='static', static_url_path='/static')
CORS(app)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'JPG', 'PNG', 'gif', 'tif'}


global src_img, pic_path, res_pic_path, message_get, pic_name, final


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def del_files(dir_path):
    # os.walk会得到dir_path下各个后代文件夹和其中的文件的三元组列表，顺序自内而外排列，
    for root, dirs, files in os.walk(dir_path, topdown=False):
        # 第一步：删除文件
        for name in files:
            os.remove(os.path.join(root, name))  # 删除文件
        # 第二步：删除空文件夹
        for name in dirs:
            os.rmdir(os.path.join(root, name))  # 删除一个空目录


# 主页
@app.route('/')
def hello():
    return render_template('main.html')


@app.errorhandler(404)
def miss404(e):
    return render_template('errors/404.html'), 404


@app.errorhandler(500)
def miss500(e):
    return render_template('errors/500.html'), 500


# 图片上传相关
@app.route('/liveExperience')
def upload_test():
    global pic_path, res_pic_path
    pic_path = 'assets/img/svg/illustrations/illustration-3.svg'
    res_pic_path = 'assets/img/svg/illustrations/illustration-7.svg'
    return render_template('live/liveExperience.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/liveExperience/upload_success', methods=['POST'])
def upload_pic():
    del_files('static/tempPics')
    img1 = request.files['photo']
    if img1 and allowed_file(img1.filename):
        img = Image.open(img1.stream)

    # 保存图片
    global pic_path, res_pic_path
    pic_path = 'tempPics/' + pic_str().create_uuid() + '.png'
    img.save('static/' + pic_path)
    res_pic_path = 'assets/img/svg/illustrations/illustration-7.svg'
    global src_img
    src_img = cv.imread('static/' + pic_path)
    src_img = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    src_img = clahe_trans(src_img)
    src_img = gamma_trans(src_img)

    return render_template('live/liveExperience.html', pic_path=pic_path, res_pic_path=res_pic_path)


@app.route('/liveExperience/upload_success', methods=['GET'])
def get_Algorithm():
    global message_get
    message_get = str(request.values.get("algorithm"))


@app.route('/liveExperience/upload_success/result')
def algorithm_process():
    global src_img, res_pic_path, pic_path, message_get, pic_name
    if message_get == 'SOBEL':
        # 边缘检测之Sobel 算子
        edges = filters.sobel(src_img)
        # 浮点型转成uint8型
        edges = img_as_ubyte(edges)
        plt.figure()
        plt.imshow(edges, plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        pic_name = 'sobel.png'
        res_pic_path = 'tempPics/' + pic_name
        plt.savefig('static/' + res_pic_path)

    elif message_get == 'OTSU':
        _, otsu_img = cv.threshold(src_img, 0, 255, cv.THRESH_OTSU)
        pic_name = 'eye_otsu.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, otsu_img)

    elif message_get == 'WATERSHED':
        # 基于直方图的二值化处理
        _, thresh = cv.threshold(src_img, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # 做开操作，是为了除去白噪声
        kernel = np.ones((3, 3), dtype=np.uint8)
        opening = cv.morphologyEx(thresh, cv.MORPH_OPEN, kernel, iterations=2)

        # 做膨胀操作，是为了让前景漫延到背景，让确定的背景出现
        sure_bg = cv.dilate(opening, kernel, iterations=2)

        # 为了求得确定的前景，也就是注水处使用距离的方法转化
        dist_transform = cv.distanceTransform(opening, cv.DIST_L2, 5)
        # 归一化所求的距离转换，转化范围是[0, 1]
        cv.normalize(dist_transform, dist_transform, 0, 1.0, cv.NORM_MINMAX)
        # 再次做二值化，得到确定的前景
        _, sure_fg = cv.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        # 得到不确定区域也就是边界所在区域，用确定的背景图减去确定的前景图
        unknow = cv.subtract(sure_bg, sure_fg)

        # 给确定的注水位置进行标上标签，背景图标为0，其他的区域由1开始按顺序进行标
        _, markers = cv.connectedComponents(sure_fg)

        # 让标签加1，这是因为在分水岭算法中，会将标签为0的区域当作边界区域（不确定区域）
        markers += 1

        # 是上面所求的不确定区域标上0
        markers[unknow == 255] = 0

        # 使用分水岭算法执行基于标记的图像分割，将图像中的对象与背景分离
        src_img = cv.cvtColor(src_img, cv.COLOR_GRAY2BGR)
        markers = cv.watershed(src_img, markers)

        # 分水岭算法得到的边界点的像素值为-1
        src_img[markers == -1] = [0, 0, 255]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        ax.set_xticks([])
        ax.set_yticks([])
        pic_name = 'watershed.png'
        res_pic_path = 'tempPics/' + pic_name
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

    elif message_get == 'FCM':
        rows, cols = src_img.shape[:2]
        pixel_count = rows * cols
        image_array = src_img.reshape(1, pixel_count)

        # 初始模糊矩阵
        init_fuzzy_mat = get_init_fuzzy_mat(pixel_count)
        # 初始聚类中心
        init_centroids = get_centroids(image_array, init_fuzzy_mat)
        fuzzy_mat, centroids, target_function = fcm(init_fuzzy_mat, init_centroids, image_array)
        label = get_label(fuzzy_mat, image_array)
        fcm_img = label.reshape(rows, cols)
        pic_name = 'fcm.png'
        res_pic_path = 'tempPics/' + pic_name
        cv.imwrite('static/' + res_pic_path, fcm_img)

    elif message_get == 'DRLSE':
        global final
        final = 0
        src_img = cv.resize(src_img, (128, 128))
        params = get_params(src_img)
        phi = find_lsf(**params)

        contours = measure.find_contours(phi, 0)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(src_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
        for n, contour in enumerate(contours):
            ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
            final = contour

        ax.fill(final[:, 1], final[:, 0], color='w')
        ax.set_xticks([])
        ax.set_yticks([])

        pic_name = 'drlse.png'
        res_pic_path = 'tempPics/' + pic_name
        print(res_pic_path)
        fig.savefig('static/' + res_pic_path, bbox_inches='tight')

        del params

    return render_template('live/show.html', pic_path=pic_path, res_pic_path=res_pic_path, temp=message_get)


# 图片下载
@app.route('/liveExperience/upload_success/result/download', methods=['GET'])
def download():
    global res_pic_path
    if request.method == "GET":
        path = 'static/tempPics'
        if path:
            return send_from_directory(path, pic_name, as_attachment=True)


@app.route("/about")
def about():
    return render_template('about.html')


@app.route('/example/')
def example_index():
    return render_template('example/example-index.html')


# 展示unet网络对不同数据集的分割效果
@app.route("/example/unet")
def example_unet():
    filename1 = 'static/csv/chest_unet.csv'
    with open(filename1) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
    filename2 = 'static/csv/eye_unet.csv'
    with open(filename2) as f:
        reader = csv.reader(f)
        data3 = next(reader)
        data4 = next(reader)
    filename3 = 'static/csv/hippocampus_unet.csv'
    with open(filename3) as f:
        reader = csv.reader(f)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/example-unet.html', data1=data1, data2=data2, data3=data3, data4=data4, data5=data5,
                           data6=data6)


# 展示胸部X光exp
@app.route('/example/chest')
def show_example_chest():
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
    return render_template('example/chest/chest-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4,
                           data5=data5)


@app.route('/example/chest/otsu')
def show_example_chest_otsu():
    filename = 'static/csv/chest_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/chest/chest-otsu-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


@app.route('/example/chest/fcm')
def show_example_chest_fcm():
    filename = 'static/csv/chest_fcm.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/chest/chest-fcm-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


@app.route('/example/chest/drlse')
def show_example_chest_drlse():
    filename = 'static/csv/chest_drlse.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/chest/chest-drlse-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


@app.route('/example/chest/unet')
def show_example_chest_unet():
    filename = 'static/csv/chest_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/chest/chest-unet-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


# 展示眼底血管exp
@app.route('/example/eye_drive')
def show_example_eye():
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    return render_template('example/eye/eye-index.html', header=header, data1=data1, data2=data2, data3=data3,
                           data4=data4,
                           data5=data5, data6=data6)


@app.route('/example/eye_drive/unet')
def show_example_eye_unet():
    filename = 'static/csv/eye_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/eye/eye-unet-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


@app.route('/example/eye_drive/otsu')
def show_example_eye_otsu():
    filename = 'static/csv/eye_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/eye/eye-otsu-show.html', data1=data1, data2=data2, data3=data3, data4=data4,
                           data5=data5,
                           data6=data6, data7=data7)


# 展示海马体exp
@app.route('/example/hippocampus')
def show_example_hippocampus():
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
    return render_template('example/hippocampus/hippocampus-index.html', header=header, data1=data1, data2=data2,
                           data3=data3, data4=data4)


@app.route('/example/hippocampus/otsu')
def show_example_hippocampus_otsu():
    filename = 'static/csv/hippocampus_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/hippocampus/hippocampus-otsu-show.html', data1=data1, data2=data2, data3=data3,
                           data4=data4,
                           data5=data5, data6=data6, data7=data7)


@app.route('/example/hippocampus/drlse')
def show_example_hippo_drlse():
    filepath = 'static/csv/hippocampus_drlse'
    data = os.listdir(filepath)
    return render_template('example/hippocampus/hippocampus-drlse-show.html', data=data)


@app.route('/example/hippocampus/unet')
def show_example_hippo_unet():
    filename = 'static/csv/hippocampus_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    return render_template('example/hippocampus/hippocampus-unet-show.html', data1=data1, data2=data2, data3=data3,
                           data4=data4,
                           data5=data5, data6=data6, data7=data7)


# exp_data
@app.route('/index_data_chest')
def line_stack_data_chest():
    data_list = {}
    filename = 'static/csv/chest_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6

    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/otsu_data_chest')
def line_stack_data_chest_otsu():
    data_list = {}
    filename = 'static/csv/chest_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/fcm_data_chest')
def line_stack_data_chest_fcm():
    data_list = {}
    filename = 'static/csv/chest_fcm.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/drlse_data_chest')
def line_stack_data_chest_drlse():
    data_list = {}
    filename = 'static/csv/chest_drlse.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/unet_data_chest')
def data_chest_unet():
    data_list = {}
    filename = 'static/csv/chest_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


# eye_exp_data
@app.route('/index_data_eye')
def data_eye():
    data_list = {}
    filename = 'static/csv/eye_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7

    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/unet_data_eye')
def data_eye_unet():
    data_list = {}
    filename = 'static/csv/eye_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/otsu_data_eye')
def data_eye_otsu():
    data_list = {}
    filename = 'static/csv/eye_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


# hippocampus_exp_data
@app.route('/index_data_hippocampus')
def data_hippocampus():
    data_list = {}
    filename = 'static/csv/hippocampus_all.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/otsu_data_hippocampus')
def data_hippocampus_otsu():
    data_list = {}
    filename = 'static/csv/hippocampus_otsu.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


@app.route('/unet_data_hippocampus')
def data_hippocampus_unet():
    data_list = {}
    filename = 'static/csv/hippocampus_unet.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        data1 = next(reader)
        data2 = next(reader)
        data3 = next(reader)
        data4 = next(reader)
        data5 = next(reader)
        data6 = next(reader)
        data7 = next(reader)
    data_list['data1'] = data1
    data_list['data2'] = data2
    data_list['data3'] = data3
    data_list['data4'] = data4
    data_list['data5'] = data5
    data_list['data6'] = data6
    data_list['data7'] = data7
    return Response(json.dumps(data_list), mimetype='application/json')


if __name__ == '__main__':
    app.run(port=1128)
