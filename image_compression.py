import io
import pickle
import zipfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.io import imread
from skimage.metrics import peak_signal_noise_ratio


def pca_compression(matrix, p):
    """Сжатие изображения с помощью PCA
    Вход: двумерная матрица (одна цветовая компонента картинки), количество компонент
    Выход: собственные векторы, проекция матрицы на новое пр-во и средние значения до центрирования
    """

    # Your code here

    # Отцентруем каждую строчку матрицы
    means = np.mean(matrix, axis=1).reshape(matrix.shape[0], 1)
    matrix_tmp = matrix - means

    # Найдем матрицу ковариации
    cov_matrix = np.cov(matrix_tmp)

    # Ищем собственные значения и собственные векторы матрицы ковариации, используйте linalg.eigh из numpy
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Посчитаем количество найденных собственных векторов
    eigenvectors_count = eigenvectors.shape[1]

    # Сортируем собственные значения в порядке убывания
    eigenvalues_idxs = np.argsort(eigenvalues)[::-1]

    # Сортируем собственные векторы согласно отсортированным собственным значениям
    # !Это все для того, чтобы мы производили проекцию в направлении максимальной дисперсии!
    eigenvectors = eigenvectors[:, eigenvalues_idxs]

    # Оставляем только p собственных векторов
    shortened_eig_vec = eigenvectors[:, :p]

    # Проекция данных на новое пространство
    res_matrix = np.dot(shortened_eig_vec.T, matrix_tmp)

    return shortened_eig_vec, res_matrix, means.flatten()


def pca_decompression(compressed):
    """Разжатие изображения
    Вход: список кортежей из собственных векторов и проекций для каждой цветовой компоненты
    Выход: разжатое изображение
    """

    result_img = []
    for i, comp in enumerate(compressed):
        # Матрично умножаем собственные векторы на проекции и прибавляем среднее значение по строкам исходной матрицы
        # !Это следует из описанного в самом начале примера!

        # Your code here

        eig_vec, projected_matrix, means = comp
        restored_channel = np.dot(eig_vec, projected_matrix) + means.reshape(means.shape[0], 1)
        result_img.append(restored_channel)

    result_img = np.clip(np.dstack(result_img), 0, 255)

    return result_img.astype(np.uint8)


def pca_visualize():
    plt.clf()
    img = imread("cat.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(3, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 5, 10, 20, 50, 100, 150, 200, 256]):
        compressed = []
        for j in range(0, 3):
            # Your code here
            channel = img[:, :, j]
            compressed.append(pca_compression(channel, p))

        restored_img = pca_decompression(compressed)

        axes[i // 3, i % 3].imshow(restored_img)
        axes[i // 3, i % 3].set_title("Компонент: {}".format(p))

    fig.savefig("pca_visualization.png")


def rgb2ycbcr(img):
    """Переход из пр-ва RGB в пр-во YCbCr
    Вход: RGB изображение
    Выход: YCbCr изображение
    """

    # Your code here

    img_tmp = img.astype(np.float64)

    R, G, B = img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    C_b = 128 - 0.1687 * R - 0.3313 * G + 0.5 * B
    C_r = 128 + 0.5 * R - 0.4187 * G - 0.0813 * B

    res = np.dstack([Y, C_b, C_r])

    return np.clip(res, 0, 255).astype(np.uint8)


def ycbcr2rgb(img):
    """Переход из пр-ва YCbCr в пр-во RGB
    Вход: YCbCr изображение
    Выход: RGB изображение
    """

    # Your code here

    img_tmp = img.astype(np.float64)

    Y, C_b, C_r = img_tmp[:, :, 0], img_tmp[:, :, 1], img_tmp[:, :, 2]

    R = Y + 1.402 * (C_r - 128)
    G = Y - 0.34414 * (C_b - 128) - 0.71414 * (C_r - 128)
    B = Y + 1.77 * (C_b - 128)

    res = np.dstack([R, G, B])

    return np.clip(res, 0, 255).astype(np.uint8)


def get_gauss_1():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here

    ycbcr_img = rgb2ycbcr(rgb_img)

    radius = 10

    filtered_ycbcr_img = np.zeros_like(ycbcr_img)
    filtered_ycbcr_img[:, :, 0] = ycbcr_img[:, :, 0]
    filtered_ycbcr_img[:, :, 1] = gaussian_filter(ycbcr_img[:, :, 1], sigma=radius)
    filtered_ycbcr_img[:, :, 2] = gaussian_filter(ycbcr_img[:, :, 2], sigma=radius)

    filtered_rgb_img = ycbcr2rgb(filtered_ycbcr_img)

    plt.imshow(filtered_rgb_img)

    plt.savefig("gauss_1.png")


def get_gauss_2():
    plt.clf()
    rgb_img = imread("Lenna.png")
    if len(rgb_img.shape) == 3:
        rgb_img = rgb_img[..., :3]

    # Your code here

    ycbcr_img = rgb2ycbcr(rgb_img)

    radius = 10

    filtered_ycbcr_img = np.zeros_like(ycbcr_img)
    filtered_ycbcr_img[:, :, 0] = gaussian_filter(ycbcr_img[:, :, 0], sigma=radius)
    filtered_ycbcr_img[:, :, 1] = ycbcr_img[:, :, 1]
    filtered_ycbcr_img[:, :, 2] = ycbcr_img[:, :, 2]

    filtered_rgb_img = ycbcr2rgb(filtered_ycbcr_img)

    plt.imshow(filtered_rgb_img)

    plt.savefig("gauss_2.png")


def downsampling(component):
    """Уменьшаем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B]
    Выход: цветовая компонента размера [A // 2, B // 2]
    """

    # Your code here

    radius = 10

    filtered_component = gaussian_filter(component, sigma=radius)

    # row_idxs = [i for i in range(0, filtered_component.shape[0], 2)]
    # column_idxs = [i for i in range(0, filtered_component.shape[1], 2)]
    #
    # res = filtered_component[row_idxs, :]
    # res = res[:, column_idxs]

    res = filtered_component[::2, ::2]

    return res


def dct(block):
    """Дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после ДКП
    """

    # Your code here

    def alpha(u):
        return 1 / np.sqrt(2) if u == 0 else 1

    def sums(u, v):

        cos_matrix = [
            np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16) for x in range(block.shape[0])
            for y in range(block.shape[1])
        ]
        cos_matrix = np.array(cos_matrix).reshape(block.shape).astype(np.float64)

        return np.sum(block.astype(np.float64) * cos_matrix)

    res = np.zeros_like(block, dtype=np.float64)

    # for u in range(block.shape[0]):
    #     for v in range(block.shape[1]):
    #         coef = 0.25 * alpha(u) * alpha(v)
    #         res[u, v] = coef * sums(u, v)

    lst_1 = [0.25 * alpha(u) * alpha(v) for u in range(block.shape[0]) for v in range(block.shape[1])]
    lst_2 = [sums(u, v) for u in range(block.shape[0]) for v in range(block.shape[1])]

    coef_matrix = np.array(lst_1).reshape(block.shape).astype(np.float64)
    sums_matrix = np.array(lst_2).reshape(block.shape).astype(np.float64)

    res = coef_matrix * sums_matrix

    return res


# Матрица квантования яркости
y_quantization_matrix = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

# Матрица квантования цвета
color_quantization_matrix = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)


def quantization(block, quantization_matrix):
    """Квантование
    Вход: блок размера 8x8 после применения ДКП; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление осуществляем с помощью np.round
    """

    # Your code here

    return np.round(block / quantization_matrix)


def own_quantization_matrix(default_quantization_matrix, q):
    """Генерация матрицы квантования по Quality Factor
    Вход: "стандартная" матрица квантования; Quality Factor
    Выход: новая матрица квантования
    Hint: если после проделанных операций какие-то элементы обнулились, то замените их единицами
    """

    assert 1 <= q <= 100

    # Your code here

    if q < 50:
        s = 5000 / q
    elif q < 100:
        s = 200 - 2 * q
    else:
        s = 1

    res = np.floor((50 + default_quantization_matrix * s) / 100)
    res[res == 0] = 1

    return res


def zigzag(block):
    """Зигзаг-сканирование
    Вход: блок размера 8x8
    Выход: список из элементов входного блока, получаемый после его обхода зигзаг-сканированием
    """

    # Your code here

    # height, width = block.shape  # 8x8
    # res = []
    # for k in range(height + width - 1):
    #     if k % 2 == 0:
    #         for i in range(k, -1, -1):
    #             j = k - i
    #             if i < height and j < width:
    #                 res.append(block[i, j])
    #     else:
    #         for j in range(k, -1, -1):
    #             i = k - j
    #             if i < height and j < width:
    #                 res.append(block[i, j])
    # return res

    # HAHAHA method :)
    zigzag_idxs = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
        (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6),
        (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    return np.array([block[i, j] for i, j in zigzag_idxs])


def compression(zigzag_list):
    """Сжатие последовательности после зигзаг-сканирования
    Вход: список после зигзаг-сканирования
    Выход: сжатый список в формате, который был приведен в качестве примера в самом начале данного пункта
    """

    # Your code here

    res = []
    chain_length = 0

    for num in zigzag_list:
        if num != 0 and not chain_length:
            res.append(num)
        elif num != 0 and chain_length:
            res.append(0)
            res.append(chain_length)
            res.append(num)
            chain_length = 0
        else:  # elif num == 0 and not chain_length or num == 0 and chain_length
            chain_length += 1

    if chain_length:
        res.append(0)
        res.append(chain_length)

    return np.array(res)


def jpeg_compression(img, quantization_matrixes):
    """JPEG-сжатие
    Вход: цветная картинка, список из 2-ух матриц квантования
    Выход: список списков со сжатыми векторами: [[compressed_y1,...], [compressed_Cb1,...], [compressed_Cr1,...]]
    """

    # Your code here

    def split_to_blocks(channel):

        blocks = []

        for i in range(0, channel.shape[0] - 7, 8):
            for j in range(0, channel.shape[1] - 7, 8):
                block = channel[i: i + 8, j: j + 8]
                block -= 128.0
                blocks.append(block)

        return blocks

    def transformation(blocks, quantization_matrix):

        res = []

        for block in blocks:
            dct_block = dct(block)
            quant_block = quantization(dct_block, quantization_matrix)
            zigzag_list = zigzag(quant_block)
            compressed_block = compression(zigzag_list)

            res.append(compressed_block)

        return res

    # Переходим из RGB в YCbCr
    ycbcr = rgb2ycbcr(img)

    # Уменьшаем цветовые компоненты
    cb_channel, cr_channel = downsampling(ycbcr[:, :, 1].copy()), downsampling(ycbcr[:, :, 2].copy())

    # Делим все компоненты на блоки 8x8 и все элементы блоков переводим из [0, 255] в [-128, 127]
    y_channel = ycbcr[:, :, 0].copy()

    blocks_y = split_to_blocks(y_channel.astype(np.float64))
    blocks_cb = split_to_blocks(cb_channel.astype(np.float64))
    blocks_cr = split_to_blocks(cr_channel.astype(np.float64))

    # Применяем ДКП, квантование, зизгаз-сканирование и сжатие
    y_quantization_matrix, color_quantization_matrix = quantization_matrixes

    y_list = transformation(blocks_y, y_quantization_matrix)
    cb_list = transformation(blocks_cb, color_quantization_matrix)
    cr_list = transformation(blocks_cr, color_quantization_matrix)

    return [y_list, cb_list, cr_list]


def inverse_compression(compressed_list):
    """Разжатие последовательности
    Вход: сжатый список
    Выход: разжатый список
    """

    # Your code here

    res = []
    idx = 0

    while idx < compressed_list.shape[0]:

        if compressed_list[idx] == 0:
            idx += 1
            res += compressed_list[idx].astype(np.uint8) * [0]
        else:
            res.append(compressed_list[idx])
        idx += 1

    return np.array(res)


def inverse_zigzag(input):
    """Обратное зигзаг-сканирование
    Вход: список элементов
    Выход: блок размера 8x8 из элементов входного списка, расставленных в матрице в порядке их следования в зигзаг-сканировании
    """

    # Your code here

    # height, width = 8, 8
    # res_block = np.zeros((height, width), dtype=np.float64)
    # idx = 0
    # for k in range(height + width - 1):
    #     if k % 2 == 0:
    #         for i in range(k, -1, -1):
    #             j = k - i
    #             if i < height and j < width:
    #                 res_block[i, j] = input[idx]
    #                 idx += 1
    #     else:
    #         for j in range(k, -1, -1):
    #             i = k - j
    #             if i < height and j < width:
    #                 res_block[i, j] = input[idx]
    #                 idx += 1
    # return res_block

    # again cinema solution
    zigzag_idxs = [
        (0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (4, 0), (3, 1), (2, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0), (6, 0), (5, 1), (4, 2), (3, 3), (2, 4), (1, 5), (0, 6),
        (0, 7), (1, 6), (2, 5), (3, 4), (4, 3), (5, 2), (6, 1), (7, 0), (7, 1), (6, 2), (5, 3), (4, 4), (3, 5), (2, 6),
        (1, 7), (2, 7), (3, 6), (4, 5), (5, 4), (6, 3), (7, 2), (7, 3), (6, 4), (5, 5), (4, 6), (3, 7), (4, 7), (5, 6),
        (6, 5), (7, 4), (7, 5), (6, 6), (5, 7), (6, 7), (7, 6), (7, 7)
    ]

    res_block = np.zeros(shape=(8, 8))

    for k in range(len(zigzag_idxs)):
        i, j = zigzag_idxs[k]
        res_block[i, j] = input[k]

    return res_block


def inverse_quantization(block, quantization_matrix):
    """Обратное квантование
    Вход: блок размера 8x8 после применения обратного зигзаг-сканирования; матрица квантования
    Выход: блок размера 8x8 после квантования. Округление не производится
    """

    # Your code here

    return block * quantization_matrix


def inverse_dct(block):
    """Обратное дискретное косинусное преобразование
    Вход: блок размера 8x8
    Выход: блок размера 8x8 после обратного ДКП. Округление осуществляем с помощью np.round
    """

    # Your code here

    def alpha(u):
        return 1 / np.sqrt(2) if u == 0 else 1

    def sums(x, y):

        cos_matrix = [
            (alpha(u) * alpha(v) * np.cos((2 * x + 1) * u * np.pi / 16) * np.cos((2 * y + 1) * v * np.pi / 16))
            for u in range(block.shape[0]) for v in range(block.shape[1])
        ]

        cos_matrix = np.array(cos_matrix).reshape(block.shape).astype(np.float64)

        return np.sum(block.astype(np.float64) * cos_matrix)

    res = np.zeros(shape=block.shape, dtype=np.float64)

    # for x in range(block.shape[0]):
    #     for y in range(block.shape[1]):
    #         res[x, y] = 0.25 * sums(x, y)

    lst = [sums(x, y) for x in range(block.shape[0]) for y in range(block.shape[1])]

    sums_matrix = np.array(lst).reshape(block.shape)

    res = 0.25 * sums_matrix

    return np.round(res)


def upsampling(component):
    """Увеличиваем цветовые компоненты в 2 раза
    Вход: цветовая компонента размера [A, B, 1]
    Выход: цветовая компонента размера [2 * A, 2 * B, 1]
    """

    # Your code here

    component_tmp = component[:, :].astype(np.float64)

    res = np.repeat(component_tmp, 2, axis=0)
    res = np.repeat(res, 2, axis=1)

    return res


def jpeg_decompression(result, result_shape, quantization_matrixes):
    """Разжатие изображения
    Вход: result список сжатых данных, размер ответа, список из 2-ух матриц квантования
    Выход: разжатое изображение
    """

    # Your code here

    def restore_blocks(compressed_blocks, quant_matrix):

        blocks = []
        for compressed in compressed_blocks:
            zigzag = inverse_compression(compressed)
            quant_block = inverse_zigzag(zigzag)
            dct_block = inverse_quantization(quant_block, quant_matrix)
            block = inverse_dct(dct_block)
            blocks.append(block)

        return blocks

    def blocks_to_component(blocks, comp_shape):

        comp_h, comp_w = comp_shape
        component = np.zeros(shape=(comp_h, comp_w), dtype=np.float64)

        idx = 0

        for i in range(0, comp_h - 7, 8):
            for j in range(0, comp_w - 7, 8):
                component[i: i + 8, j: j + 8] = blocks[idx]
                component[i: i + 8, j: j + 8] += 128.0

                idx += 1

        return component

    compressed_y, compressed_cb, compressed_cr = result
    y_quantization_matrix, color_quantization_matrix = quantization_matrixes

    if len(result_shape) == 3:
        height, width, _ = result_shape
    else:
        height, width = result_shape

    blocks_y = restore_blocks(compressed_y, y_quantization_matrix)
    blocks_cb = restore_blocks(compressed_cb, color_quantization_matrix)
    blocks_cr = restore_blocks(compressed_cr, color_quantization_matrix)

    y_comp = blocks_to_component(blocks_y, (height, width))
    cb_comp = blocks_to_component(blocks_cb, (height // 2, width // 2))
    cr_comp = blocks_to_component(blocks_cr, (height // 2, width // 2))

    cb_upsampled = upsampling(cb_comp)
    cr_upsampled = upsampling(cr_comp)

    y_comp = np.clip(y_comp, 0, 255).astype(np.uint8)
    cb_upsampled = np.clip(cb_upsampled, 0, 255).astype(np.uint8)
    cr_upsampled = np.clip(cr_upsampled, 0, 255).astype(np.uint8)

    ycbcr_img = np.dstack([y_comp, cb_upsampled, cr_upsampled])

    rgb_result = ycbcr2rgb(ycbcr_img)

    return rgb_result


def jpeg_visualize():
    plt.clf()
    img = imread("Lenna.png")
    if len(img.shape) == 3:
        img = img[..., :3]
    fig, axes = plt.subplots(2, 3)
    fig.set_figwidth(12)
    fig.set_figheight(12)

    for i, p in enumerate([1, 10, 20, 50, 80, 100]):
        # Your code here

        # print(f'Quality Factor = {p}')
        y_quant = own_quantization_matrix(y_quantization_matrix, p)
        color_quant = own_quantization_matrix(color_quantization_matrix, p)
        quantization_matrixes = [y_quant, color_quant]

        compressed_result = jpeg_compression(img, quantization_matrixes)
        decompressed_img = jpeg_decompression(compressed_result, img.shape[:2], quantization_matrixes)

        axes[i // 3, i % 3].imshow(decompressed_img)
        axes[i // 3, i % 3].set_title("Quality Factor: {}".format(p))

    fig.savefig("jpeg_visualization.png")


def get_deflated_bytesize(data):
    raw_data = pickle.dumps(data)
    with io.BytesIO() as buf:
        with (
            zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zipf,
            zipf.open("data", mode="w") as handle,
        ):
            handle.write(raw_data)
            handle.flush()
            handle.close()
            zipf.close()
        buf.flush()
        return buf.getbuffer().nbytes


def compression_pipeline(img, c_type, param=1):
    """Pipeline для PCA и JPEG
    Вход: исходное изображение; название метода - 'pca', 'jpeg';
    param - кол-во компонент в случае PCA, и Quality Factor для JPEG
    Выход: изображение; количество бит на пиксель
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    if c_type.lower() == "jpeg":
        y_quantization = own_quantization_matrix(y_quantization_matrix, param)
        color_quantization = own_quantization_matrix(color_quantization_matrix, param)
        matrixes = [y_quantization, color_quantization]

        compressed = jpeg_compression(img, matrixes)
        img = jpeg_decompression(compressed, img.shape, matrixes)
        compressed_size = get_deflated_bytesize(compressed)

    elif c_type.lower() == "pca":
        compressed = [
            pca_compression(c.copy(), param)
            for c in img.transpose(2, 0, 1).astype(np.float64)
        ]

        img = pca_decompression(compressed)
        compressed_size = sum(d.nbytes for c in compressed for d in c)

    raw_size = img.nbytes

    return img, compressed_size / raw_size


def calc_metrics(img_path, c_type, param_list):
    """Подсчет PSNR и Compression Ratio для PCA и JPEG. Построение графиков
    Вход: пусть до изображения; тип сжатия; список параметров: кол-во компонент в случае PCA, и Quality Factor для JPEG
    """

    assert c_type.lower() == "jpeg" or c_type.lower() == "pca"

    img = imread(img_path)
    if len(img.shape) == 3:
        img = img[..., :3]

    outputs = []
    for param in param_list:
        outputs.append(compression_pipeline(img.copy(), c_type, param))

    psnr = [peak_signal_noise_ratio(img, output[0]) for output in outputs]
    ratio = [output[1] for output in outputs]

    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(20)
    fig.set_figheight(5)

    ax1.set_title("Quality Factor vs PSNR for {}".format(c_type.upper()))
    ax1.plot(param_list, psnr, "tab:orange")
    ax1.set_ylim(13, 64)
    ax1.set_xlabel("Quality Factor")
    ax1.set_ylabel("PSNR")

    ax2.set_title("PSNR vs Compression Ratio for {}".format(c_type.upper()))
    ax2.plot(psnr, ratio, "tab:red")
    ax2.set_xlim(13, 30)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("PSNR")
    ax2.set_ylabel("Compression Ratio")
    return fig


def get_pca_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "pca", [1, 5, 10, 20, 50, 100, 150, 200, 256])
    fig.savefig("pca_metrics_graph.png")


def get_jpeg_metrics_graph():
    plt.clf()
    fig = calc_metrics("Lenna.png", "jpeg", [1, 10, 20, 50, 80, 100])
    fig.savefig("jpeg_metrics_graph.png")


if __name__ == "__main__":
    pca_visualize()
    get_gauss_1()
    get_gauss_2()
    jpeg_visualize()
    get_pca_metrics_graph()
    get_jpeg_metrics_graph()
