import numpy as np
from tifffile import imwrite, memmap
import matplotlib.pyplot as plt
from matplotlib import colors
from skimage.filters import median, gaussian


def phasor(image_stack):
    """
    :param image_stack: is a file with spectral mxm images to calculate the fast fourier transform from
    numpy library.
    :return: g: is mxm image with the real part of the fft.
    :return: s: is mxm imaginary with the real part of the fft.
    :return: md: numpy.ndarray  It is the modulus obtain with Euclidean Distance.
    :return: ph: is the phase between g ans s in degrees.
    """

    data = np.fft.fft(image_stack, axis=0)
    dc = data[0].real
    # change the zeros to the img average
    dc = np.where(dc != 0, dc, int(np.mean(dc)))
    g = data[1].real
    g /= -dc
    s = data[1].imag
    s /= -dc

    md = np.sqrt(g ** 2 + s ** 2)
    ph = np.angle(data[1], deg=True)

    return g, s, md, ph, dc


def generate_file(filename, gsa):
    """
    :param filename: Type string characters. The name of the file to be written, with the extension ome.tiff
    :param gsa: Type n-dimensional array holding the data to be stored. It usually has the mxm images of
    g,s and the average image.
    :return file: The created file storing the data. If the filename extension was ome.tiff the file is an
    ome.tiff format.
    """

    imwrite(filename, data=gsa)
    file = memmap(filename)
    file.flush()

    return file


def concat_d2(im):
    """
    :param im: stack image with the images to be concatenated. It is a specific 2x2 concatenation.
    :return: im_concat it is an image stack with the concatenated images.
    """

    im_concat = []

    # Caso 1 donde es na imagen de 4 partes y un solo canal
    if len(im.shape) == 3 and im.shape[0] == 4:
        d = 1024
        l = int(d * 0.05)
        m = int(d / 2 + l)
        n = int(d / 2 - l)

        im_concat = np.zeros([d, d])

        # Concatenate 4 images excluding the interjection
        im_concat[0:n, 0:n] = im[0][0:n, 0:n]
        im_concat[0:n, m:d] = im[1][0:n, 2 * l:m]
        im_concat[m:d, 0:n] = im[2][2 * l:m, 0:n]
        im_concat[m:d, m:d] = im[3][2 * l:m, 2 * l:m]

        # Average the common zones and put them in the final concatenated image
        # vertical
        imp1 = (im[0][0:n, n:m] + im[1][0:n, 0:2 * l]) / 2
        imp2 = (im[2][2 * l:m, n:m] + im[3][2 * l:m, 0:2 * l]) / 2
        # horizontal
        imp3 = (im[0][n:m, 0:n] + im[2][0:2 * l, 0:n]) / 2
        imp4 = (im[1][n:m, 2 * l:m] + im[3][0:2 * l, 2 * l:m]) / 2
        # centre
        imp_centro = (im[0][n:m, n:m] + im[1][0:2 * l, n:m] + im[2][0:2 * l, n:m] +
                      im[3][0:2 * l, 0:2 * l]) / 4

        # Add the last part to the concatenated image
        im_concat[n:m, n:m] = imp_centro
        im_concat[0:n, n:m] = imp1
        im_concat[m:d, n:m] = imp2
        im_concat[n:m, 0:n] = imp3
        im_concat[n:m, m:d] = imp4

    # Caso 2 donde es una imagen de 4 partes y mas de un canal
    elif len(im.shape) == 4 and im.shape[0] == 4:
        d = 1024
        t = im.shape[1]
        s = im.shape[2]
        l = int(d * 0.05)
        m = int(d / 2 + l)
        n = int(d / 2 - l)

        im_concat = np.zeros([t, d, d])
        # verticales
        imp1 = np.zeros([s, n, 2 * l])
        imp2 = np.zeros([s, n, 2 * l])
        # horizontales
        imp3 = np.zeros([s, 2 * l, n])
        imp4 = np.zeros([s, 2 * l, n])
        imp_centro = np.zeros([s, 2 * l, 2 * l])

        for i in range(0, t):
            # Concatenate 4 images excluding the interjection
            im_concat[i][0:n, 0:n] = im[0][i][0:n, 0:n]
            im_concat[i][0:n, m:d] = im[1][i][0:n, 2 * l:m]
            im_concat[i][m:d, 0:n] = im[2][i][2 * l:m, 0:n]
            im_concat[i][m:d, m:d] = im[3][i][2 * l:m, 2 * l:m]

            # Average the common zones and put them in the final concatenated image
            # vertical
            imp1[i] = (im[0][i][0:n, n:m] + im[1][i][0:n, 0:2 * l]) / 2
            imp2[i] = (im[2][i][2 * l:m, n:m] + im[3][i][2 * l:m, 0:2 * l]) / 2
            # horizontal
            imp3[i] = (im[0][i][n:m, 0:n] + im[2][i][0:2 * l, 0:n]) / 2
            imp4[i] = (im[1][i][n:m, 2 * l:m] + im[3][i][0:2 * l, 2 * l:m]) / 2
            # centre
            imp_centro[i] = (im[0][i][n:m, n:m] + im[1][i][0:2 * l, n:m] + im[2][i][0:2 * l, n:m] +
                             im[3][i][0:2 * l, 0:2 * l]) / 4

            # Add the last part to the concatenated image
            im_concat[i][n:m, n:m] = imp_centro[i]
            im_concat[i][0:n, n:m] = imp1[i]
            im_concat[i][m:d, n:m] = imp2[i]
            im_concat[i][n:m, 0:n] = imp3[i]
            im_concat[i][n:m, m:d] = imp4[i]

    else:
        print('Image stack dimension is not 3 or 4. So it returns null.')

    return im_concat


def ndmedian(im, filttime=0):
    """
    :param im: ndarray usually an image to be filtered.
    :param filttime: numbers of time to be filtered im.
    :return: ndarray with the filtered image.
    """
    imfilt = im
    i = 0
    while i < filttime:
        imfilt = median(im)
        i = i + 1
    return imfilt


def phasor_plot(img_mean, g, s, icut, title=['Phasor']):
    """
    :param img_mean: image stack with all the average images related to each phasor nxn dimension.
    :param g: nxn dimension image.
    :param s: nxn dimension image.
    :param icut: array lenth numbers of g images contains the cut intensity for related to each avg img.
    :param title: (optional) the title of each phasor
    :return: the phasor figure and x and y arrays containing the G and S values.
    """
    num_phasors = len(img_mean)
    x = []
    y = []
    for k in range(0, len(g)):
        x.append([])
        y.append([])
        for i in range(0, len(g[k])):
            for j in range(0, len(g[k][0])):
                if img_mean[k][i][j] > icut[k]:
                    x[k].append(g[k][i][j])
                    y[k].append(s[k][i][j])

    # built the figure inner and outer circle and the 45 degrees lines in the plot
    x1 = np.linspace(start=-1, stop=1, num=500)
    y_positive = lambda x1: np.sqrt(1 - x1 ** 2)
    y_negative = lambda x1: -np.sqrt(1 - x1 ** 2)
    x2 = np.linspace(start=-0.5, stop=0.5, num=500)
    y_positive2 = lambda x2: np.sqrt(0.5 ** 2 - x2 ** 2)
    y_negative2 = lambda x2: -np.sqrt(0.5 ** 2 - x2 ** 2)
    x3 = np.linspace(start=-1, stop=1, num=30)
    x4 = np.linspace(start=-0.7, stop=0.7, num=30)

    # create the figures with all the phasors in each axes or create only one phasor
    if num_phasors > 1:
        fig, ax = plt.subplots(1, num_phasors, figsize=(13, 4))
        fig.suptitle('Phasor')
        for i in range(num_phasors):
            ax[i].hist2d(x[i], y[i], bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
            ax[i].plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
            ax[i].plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
            ax[i].plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
            ax[i].plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
            ax[i].scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
            ax[i].scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
            ax[i].scatter(x4, x4, marker='_', color='darkgoldenrod')
            ax[i].scatter(x4, -x4, marker='_', color='darkgoldenrod')
            ax[i].set_title(title[i])
    else:
        fig = plt.figure(1)
        plt.hist2d(x[0], y[0], bins=256, cmap="RdYlGn_r", norm=colors.LogNorm(), range=[[-1, 1], [-1, 1]])
        plt.title(title[0])
        plt.plot(x1, list(map(y_positive, x1)), color='darkgoldenrod')
        plt.plot(x1, list(map(y_negative, x1)), color='darkgoldenrod')
        plt.plot(x2, list(map(y_positive2, x2)), color='darkgoldenrod')
        plt.plot(x2, list(map(y_negative2, x2)), color='darkgoldenrod')
        plt.scatter(x3, [0] * len(x3), marker='_', color='darkgoldenrod')
        plt.scatter([0] * len(x3), x3, marker='|', color='darkgoldenrod')
        plt.scatter(x4, x4, marker='_', color='darkgoldenrod')
        plt.scatter(x4, -x4, marker='_', color='darkgoldenrod')

    return fig, x, y
