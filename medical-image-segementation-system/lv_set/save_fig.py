import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

final = 0
fig1 = plt.figure(1)
fig2 = plt.figure(2)


def save_fig(phi, img, label_img, name):
    global final

    contours = measure.find_contours(phi, 0)
    drlse_img = np.ones((img.shape[1], img.shape[0]), np.uint8)

    ax1 = fig1.add_subplot(121)
    ax1.imshow(img, interpolation='nearest', cmap=plt.get_cmap('gray'))

    for n, contour in enumerate(contours):
        ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
        final = contour

    ax1.fill(final[:, 1], final[:, 0], color='w')
    ax1.set_xticks([])
    ax1.set_yticks([])

    ax2 = fig1.add_subplot(122)
    ax2.imshow(label_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig1.savefig('../resImg/DRLSE_res_20/' + name + '.jpg', bbox_inches='tight')
    fig1.clear()

    ax3 = fig2.add_subplot(111)
    ax3.imshow(drlse_img, interpolation='nearest', cmap=plt.get_cmap('gray'))
    ax3.fill(final[:, 0], final[:, 1], color='w')
    plt.axis('off')
    fig = plt.gcf()
    fig.set_size_inches(img.shape[0]/96, img.shape[1]/96)  # dpi = 96
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig('chest1.png', transparent=True, dpi=96, pad_inches=0)
