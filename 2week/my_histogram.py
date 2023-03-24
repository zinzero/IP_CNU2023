import numpy as np
import matplotlib.pyplot as plt
import cv2

def point_processing(src,type='original'):

    dst = np.zeros(src.shape, dtype=np.uint8)

    if type == 'original':
        dst = src.copy()

    elif type == 'darken':
        "x - 128"
        # uint8이었던 원본 이미지를 float로 바꿔서 계산
        # Underflow가 발생한 부분에 대해서는(값이 음수인 부분) 0으로 처리
        # 그 후 다시 np.uint8로 변경
        dst = src.astype(np.float32) - 128
        dst[dst < 0] = 0
        dst = dst.astype(np.uint8)

    elif type == 'lower_contrast':

        # uint8형으로 표현하고 싶은 경우
        dst = src / 3
        dst = np.round(dst).astype(np.uint8)

        # float형으로 표현하고 싶은 경우
        # dst = (src / 255) / 2

        # 이렇게 하면 안됨
        # dst = src / 2
        # dst.dtype -> float 자료형
        # float 자료형은 0 ~ 1 사이의 값만 유의미함

    elif type == 'non_linear_lower_contrast':

        dst = ((src / 255) ** (1 / 3)) * 255  # dst.dtype : float

        # uint형으로 표현하고 싶은 경우
        dst = np.round(dst).astype(np.uint8)

        # float 타입으로 이미지를 생성하고 싶은 경우
        # 0 ~ 1 사이로 normalization
        #dst = dst / dst.max()


    elif type == 'invert':
        dst = 255 - src

    elif type == 'lighten':

        dst = src.astype(np.float32) + 128
        dst[dst > 255] = 255
        dst = dst.astype(np.uint8)

    elif type == 'raise_contrast':

        dst = src.astype(np.float32) * 2
        dst[dst > 255] = 255
        dst = dst.astype(np.uint8)

    elif type == 'non_linear_raise_contrast':

        dst = ((src / 255) ** 2) * 255
        # uint8로 표현하고 싶은 경우
        dst = np.round(dst).astype(np.uint8)

        # float 타입으로 이미지를 생성하고 싶은 경우
        # 0 ~ 1 사이로 normalization
        #dst = dst / dst.max()


    return dst

def my_cal_Hist(img):
    """
    Argument info
    img: gray scale image (H x W)

    variable info
    hist : 이미지의 픽셀 값들의 빈도수를 세는 1차원 배열
    hist의 index는 이미지의 픽셀 값을 의미 i.e) pixel 값 18 == index 18

    return info
    hist: 입력 이미지의 각 픽셀 빈도수를 나타내는 배열
    """
    h, w = img.shape
    # 주어진 이미지의 가진 1차원 배열 생성
    hist = np.zeros(256, dtype=np.int)

    # histogram bin
    for row in range(h):
        for col in range(w):
            intensity = img[row, col]
            hist[intensity] += 1

    return hist

def save_img(path, img):
    return cv2.imwrite(path, img)


def plot_histogram(src):

    # histogram 계산
    hist = my_cal_Hist(src)
    bin_x = np.arange(len(hist))
    plt.bar(bin_x, hist, width=0.8, color='g')
    plt.title('my_histogram')
    plt.xlabel('pixel intensity')
    plt.ylabel('pixel frequency')
    plt.show()



def main():

    src = cv2.imread('./fruits.jpg', cv2.IMREAD_GRAYSCALE)
    dst = point_processing(src, 'non_linear_raise_contrast')

    # Histogram 시각화
    plot_histogram(dst)

    return



if __name__ == '__main__':
    main()