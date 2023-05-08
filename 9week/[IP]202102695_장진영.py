import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_threshold_by_within_variance(intensity, p):

    """
    :param intensity: pixel 값 0 ~ 255 범위를 갖는 배열
    :param p: 상대도수 값
    :return: k: 최적의 threshold 값
    """

    ########################################################
    # TODO
    # TODO otsu_method 완성
    # TODO  1. within-class variance를 이용한 방법
    # TODO  교수님 이론 PPT 22 page 참고
    ########################################################

    # ???
    # i 까지의 p의 합
    q1 = np.zeros((256,))
    q2 = np.zeros((256,))
    for i in range(0, 256):
        q1[i] = p[i]
        q2[i] = 1 - q1[i]


    # i 까지의 평균
    m1 = np.zeros((256,))
    m2 = np.zeros((256,))
    for i in range(0, 256):
        if q1[i] != 0:
            m1[i] = (p[i] * intensity[i]) / q1[i]
        else:
            m1[i] = 0
        for j in range(i + 1, 256):
            if q2[j] != 0:
                m2[j] = (p[j] * intensity[j]) / q2[j]
            else:
                m2[j] = 0

    # m1 = [0 for _ in range(256)]
    # m2 = [0 for _ in range(256)]
    # for i in range(0, 256):
    #     m1[i] = (p[i] * intensity[i]) / q1[i]
    #     for j in range(i, 256):
    #         m1[i] = (p[j] * intensity[j]) / q1[j]

    # 분산
    sigma1 = np.zeros((256,))
    sigma2 = np.zeros((256,))
    for i in range(0, 256):
        if q1[i] != 0:
            sigma1[i] = ((intensity[i] ** 2) * p[i] - (m1[i] ** 2)) / q1[i]
        else:
            sigma1[i] = 0
        for j in range(i + 1, 256):
            if q2[j] != 0:
                sigma2[j] = ((intensity[j] ** 2) * p[j] - (m2[j] ** 2)) / q2[j]
            else:
                sigma2[j] = 0

    # sigma1 = [0 for _ in range(256)]
    # sigma2 = [0 for _ in range(256)]
    # for i in range(0, 256):
    #     sigma1[i] = ((intensity[i] ** 2) * p[i] - (m1[i] ** 2)) / q1[i]
    #     for j in range(i, 256):
    #         sigma2[i] = ((intensity[i] ** 2) * p[i] - (m2[j] ** 2)) / q2[j]


    sigma = [0 for _ in range(256)]
    for i in range(0, 256):
        sigma[i] = q1[i] * sigma1[i] + q2[i] * sigma2[i]

    k = np.argmin(sigma)

    return k

def get_threshold_by_inter_variance(p):

    """
    :param p: 상대도수 값
    :return: k: 최적의 threshold 값
    """

    ########################################################
    # TODO
    # TODO otsu_method 완성
    # TODO  2. inter-class variance를 이용한 방법
    # TODO  Moving average를 이용하여 구현
    # TODO  교수님 이론 PPT 26 page 참고
    ########################################################

    p += 1e-7  # q1과 q2가 0일때 나눗셈을 진행할 경우 오류를 막기 위함

    # ???
    q1 = [0 for _ in range(256)]
    q1.append(p[0])
    for i in range(0, 255):
        q1[i + 1] = q1[i] + p[i + 1]

    m1 = [0 for _ in range(256)]
    for i in range(0, 255):
        m1[i + 1] = (q1[i] * m1[i] + (i + 1) * p[i + 1]) / q1[i + 1]

    m2 = [0 for _ in range(256)]
    for i in range(1, 256):
        m2[0] += p[i] * i

    m2[0] = m2[0] / (1 - q1[0])

    for i in range(0, 255):
        m2[i + 1] = ((1 - q1[i]) * m2[i] - (i + 1) * p[i + 1]) / (1 - q1[i + 1])

    sigma = [0 for _ in range(256)]
    for i in range(0, 256):
        sigma[i] = q1[i] * (1 - q1[i]) * ((m1[i] - m2[i]) ** 2)

    k = np.argmin(sigma)

    return k


def get_hist(src, mask):

    """
    :param src: gray scale 이미지
    :param mask: masking을 하기 위한 값
    :return:
    """

    #######################################################################
    # TODO mask를 적용한 히스토그램 완성
    # TODO mask 값이 0인 영역은 픽셀의 빈도수를 세지 않음
    # TODO histogram을 생성해 주는 내장함수 사용금지. np.histogram, cv2.calHist
    #######################################################################
    hist = np.zeros((256,))

    # ???
    (h, w) = src.shape

    for row in range(h):
        for col in range(w):
            if mask[row, col] != 0:
                intensity = src[row, col]
                hist[intensity] += 1

    return hist


def threshold(src, threshold, mask):
    """
    :param src: gray scale 이미지
    :param threshold: threshold 값
    :param mask: masking을 하기 위한 값
    :return:
    """

    ########################################################
    # TODO threshold 값을 이용한 이미지 값 채우기
    # TODO 0 < src <= threshold 이면 255로 채움
    # TODO mask의 값이 0인 좌표에 대해서는 값을 0으로 채움
    # TODO 이외의 영역은 모두 0으로 채움
    # TODO cv2.threshold 사용금지
    ########################################################

    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)

    # ???
    for row in range(h):
        for col in range(w):
            if mask[row, col] == 0:
                dst[row, col] = 0
            else:
                if 0 < src[row, col] <= threshold:
                    dst[row, col] = 255
                else:
                    dst[row, col] = 0

    return dst


def otsu_method(src, mask):

    """
    :param src: 원본 이미지와 mask를 곱한 이미지
    :param mask: 0 or 1의 값을 갖음
            histogram과 threshold 과정시 mask 값이 0에 해당하는 index는 처리를 하지 않기 위함

    변수 정보
    hist: 위의 src에 대하여 masking을 적용한 히스토그램을 구한 배열
    intensity: 0 ~ 255의 값을 갖는 배열
    p: 상대 도수 (히스토그램의 정규화)
    k1: within variance 방식으로 구한 최적의 threshold 값
    k2: inter variance 방식으로 구한 최적의 threshold 값

    :return: 2가지 방식으로 thresholding된 결과 이미지들 (dst1, dst2)
    """

    hist = get_hist(src, mask)
    hist = hist.astype((np.int32))
    intensity = np.array([i for i in range(256)])

    ########################################################
    # TODO 상대도수 p 구하기
    # TODO 교수님 이론 PPT 17 page -> p_{i}에 해당
    ########################################################
    # p = ???
    p = hist / np.sum(hist)
    ########################################################
    # TODO otsu_method 완성
    # TODO  1. within-class variance를 이용한 방법
    # TODO      (get_threshold_by_within_variance 함수 사용)
    # TODO  2. between-class variance를 이용한 방법
    # TODO      (get_threshold_by_inter_variance 함수 사용)
    ########################################################

    k1 = get_threshold_by_within_variance(intensity, p)
    k2 = get_threshold_by_inter_variance(p)

    # k1과 k2가 같아야 한다.
    # 같지 않으면 실행 종료
    print("k1 : ", k1)
    print("k2 : ", k2)
    assert k1 == k2

    dst1 = threshold(src, k1, mask)
    dst2 = threshold(src, k2, mask)

    ########################################################
    # TODO Bimodal histogram 완성
    # TODO 2개의 peak에 해당하는 픽셀 값에 점 찍기
    # TODO 보고서에 결과 이미지 첨부
    ########################################################
    # Bi-modal Distribution
    plt.plot(intensity, hist)
    # plt.plot(???, ???, color='red', marker='o', markersize=6)
    # plt.plot(???, ???, color='red', marker='o', markersize=6)
    plt.plot(np.argmax(hist[:k1]), hist[np.argmax(hist[:k1])], color='red', marker='o', markersize=6)
    plt.plot(np.argmax(hist[k1 + 1:]), hist[np.argmax(hist[k1 + 1:])], color='red', marker='o', markersize=6)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    plt.title('Interest region histogram')
    plt.show()

    return dst1, dst2


def main():
    meat = cv2.imread('meat.png', cv2.IMREAD_GRAYSCALE)
    # mask 값을 0 또는 1로 만들기 위해 255로 나누어줌
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE) / 255

    # meat.png에서 관심 영역 해당하는 부분만 따로 추출.
    src = (meat * mask).astype(np.uint8)

    # 추출된 이미지 확인.
    cv2.imshow('original', meat)
    cv2.imshow('interest src', src)
    cv2.waitKey()
    cv2.destroyAllWindows()

    # 관심 영역 해당하는 이미지에 대하여 otsu's method 적용
    dst1, dst2 = otsu_method(src, mask)

    # True 값이 나와야함
    print("2가지 방식 결과 비교 : {}".format(np.array_equal(dst1, dst2)))

    # 원본 이미지에 적용하기
    final1 = cv2.add(meat, dst1)
    final2 = cv2.add(meat, dst1)

    # 본인 학번 적기
    cv2.imshow('202102695 within_variance dst', dst1)
    cv2.imshow('202102695 inter_variance dst', dst2)
    cv2.imshow('202102695 final1', final1)
    cv2.imshow('202102695 final2', final2)

    cv2.waitKey()
    cv2.destroyAllWindows()

    # 보고서 첨부용
    cv2.imwrite('dst_by_within_variance.png', dst1)
    cv2.imwrite('dst_by_inter_variance.png', dst2)
    cv2.imwrite('final_by_within_variance.png', final1)
    cv2.imwrite('final_by_inter_variance.png', final2)

    return


if __name__ == '__main__':
    main()