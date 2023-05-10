import cv2
import numpy as np

def my_padding(src, filter):
    (h, w) = src.shape
    (h_pad, w_pad) = filter.shape
    h_pad = h_pad // 2
    w_pad = w_pad // 2
    padding_img = np.zeros((h+h_pad*2, w+w_pad*2))
    padding_img[h_pad:h+h_pad, w_pad:w+w_pad] = src
    return padding_img

def my_filtering(src, filter):
    (h, w) = src.shape
    (m_h, m_w) = filter.shape
    pad_img =my_padding(src, filter)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * filter)
    return dst

def convert_uint8(img):
    #이미지 출력을 위해서 타입을 변경 수행
    return ((img - np.min(img)) / np.max(img - np.min(img)) * 255).astype(np.uint8)

def get_DoG_filter():
    ###################################################
    # TODO                                            #
    # TODO DoG mask 완성                                    #
    # TODO DoG의 경우 과제가 진행중이기에 저장된 배열을 가지고 와서
    # TODO 불러오는 형식으로 진행함.
    # TODO 함수를 고칠 필요는 전혀 없음.
    ###################################################

    DoG_x = np.load('DoG_x.npy')
    DoG_y = np.load('DoG_y.npy')

    return DoG_x, DoG_y

def calcMagnitude(Ix, Iy):
    ###########################################
    # TODO                                    #
    # calcMagnitude 완성                      #
    # magnitude : ix와 iy의 magnitude         #
    ###########################################
    # Ix와 Iy의 magnitude를 계산
    # magnitude = None
    magnitude = np.sqrt((Ix ** 2) + (Iy ** 2))
    print("mag : ", magnitude)
    return magnitude

def calcAngle(Ix, Iy):
    #######################################
    # TODO                                #
    # calcAngle 완성                      #
    # angle     : ix와 iy의 angle         #
    #######################################

    (h, w) = Ix.shape
    angle = np.zeros((h, w))

    for i in range(h):
        for j in range(w):
            if Ix[i, j] == 0:
                if Iy[i, j] < 0:
                    angle[i, j] = -90
                elif Iy[i, j] > 0:
                    angle[i, j] = 90
                else:
                    angle[i, j] = 0
            else:
                angle[i, j] = np.rad2deg(np.arctan(Iy[i, j] / Ix[i, j]))

    return angle

def pixel_bilinear_coordinate(src, pixel_coordinate):
    ####################################################################################
    # TODO                                                                             #
    # TODO Pixel-Bilinear Interpolation 완성
    # TODO 진행과정
    # TODO 저번 실습을 참고로 픽셀 위치를 기반으로 주변 픽셀을 가져와서 Interpolation을 구현
    ####################################################################################

    h, w = src.shape

    # 주변 픽셀 위치 4개를 가져옴.
    # 가져오는 방식은 저번주 실습을 참고하여 가져오는 것을 추천.
    # y_up = None
    # y_down = None
    # x_left = None
    # x_right = None
    y_up = int(pixel_coordinate[0])
    y_down = min(int(pixel_coordinate[0] + 1), h - 1)
    x_left = int(pixel_coordinate[1])
    x_right = min(int(pixel_coordinate[1] + 1), w - 1)

    # x 비율, y 비율을 계산하는 코드
    # 저번 실습 자료 참고.
    # t = None
    # s = None
    t = pixel_coordinate[0] - y_up
    s = pixel_coordinate[1] - x_left

    # Bilinear Interpolation 구현 부분
    # 저번 실습 자료 참고.
    # intensity = None
    intensity = ((1 - s) * (1 - t) * src[y_up, x_left]) \
                + (s * (1 - t) * src[y_up, x_right]) \
                + ((1 - s) * t * src[y_down, x_left]) \
                + (s * t * src[y_down, x_right])

    return intensity

def non_maximum_supression_three_size(magnitude, angle):
    ####################################################################################
    # TODO
    # TODO non_maximum_supression
    # TODO largest_magnitude: non_maximum_supression 결과(가장 강한 edge만 남김)         #
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))
    for row in range(1, h - 1):
        for col in range(1, w - 1):
            degree = angle[row, col]

            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree < 45:
                rate = np.tan(np.deg2rad(degree))
                left_pixel_coordinate = (row + rate, col + 1)
                right_pixel_coordinate = (row - rate, col - 1)
                left_magnitude = pixel_bilinear_coordinate(magnitude, left_pixel_coordinate)
                right_magnitude = pixel_bilinear_coordinate(magnitude, right_pixel_coordinate)
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif 45 <= degree <= 90:
                rate = np.tan(np.deg2rad(90 - degree))  # cotan = 1/tan
                up_pixel_coordinate = (row + 1, col + rate)
                down_pixel_coordinate = (row - 1, col - rate)
                up_magnitude = pixel_bilinear_coordinate(magnitude, up_pixel_coordinate)
                down_magnitude = pixel_bilinear_coordinate(magnitude, down_pixel_coordinate)
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -45 <= degree < 0:
                rate = -np.tan(np.deg2rad(degree))
                left_pixel_coordinate = (row - rate, col + 1)
                right_pixel_coordinate = (row + rate, col - 1)
                left_magnitude = pixel_bilinear_coordinate(magnitude, left_pixel_coordinate)
                right_magnitude = pixel_bilinear_coordinate(magnitude, right_pixel_coordinate)
                if magnitude[row, col] == max(left_magnitude, magnitude[row, col], right_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            elif -90 <= degree < -45:
                rate = -np.tan(np.deg2rad(90 - degree))
                up_pixel_coordinate = (row - 1, col + rate)
                down_pixel_coordinate = (row + 1, col - rate)
                up_magnitude = pixel_bilinear_coordinate(magnitude, up_pixel_coordinate)
                down_magnitude = pixel_bilinear_coordinate(magnitude, down_pixel_coordinate)
                if magnitude[row, col] == max(up_magnitude, magnitude[row, col], down_magnitude):
                    largest_magnitude[row, col] = magnitude[row, col]

            else:
                print(row, col, 'error!  degree :', degree)

    return largest_magnitude

def non_maximum_supression_five_size(magnitude, angle, step=0.5):
    ####################################################################################
    # TODO
    # TODO non_maximum_supression 완성 5x5 영역
    # TODO largest_magnitude: non_maximum_supression 결과(가장 강한 edge만 남김)
    ####################################################################################
    (h, w) = magnitude.shape
    # angle의 범위 : -90 ~ 90
    largest_magnitude = np.zeros((h, w))

    for row in range(2, h - 2):
        for col in range(2, w - 2):
            degree = angle[row, col]

            # gradient의 degree는 edge와 수직방향이다.
            if 0 <= degree < 45:
                for i in np.arange(-2, 2.5, step):
                    if i == 0:
                        continue
                    else:
                        rate = np.tan(np.deg2rad(degree)) * i
                        pixel_coordinate = (row + rate, col + i)
                        pixel_magnitude = pixel_bilinear_coordinate(magnitude, pixel_coordinate)
                        if magnitude[row, col] == max(pixel_magnitude, magnitude[row, col]):
                            largest_magnitude[row, col] = magnitude[row, col]
                        else:
                            largest_magnitude[row, col] = 0
                            break

            elif 45 <= degree <= 90:
                for i in np.arange(-2, 2.5, step):
                    if i == 0:
                        continue
                    else:
                        rate = np.tan(np.deg2rad(90 - degree)) * i
                        pixel_coordinate = (row + i, col + rate)
                        pixel_magnitude = pixel_bilinear_coordinate(magnitude, pixel_coordinate)
                        if magnitude[row, col] == max(pixel_magnitude, magnitude[row, col]):
                            largest_magnitude[row, col] = magnitude[row, col]
                        else:
                            largest_magnitude[row, col] = 0
                            break

            elif -45 <= degree < 0:
                for i in np.arange(-2, 2.5, step):
                    if i == 0:
                        continue
                    else:
                        rate = -np.tan(np.deg2rad(degree)) * i
                        pixel_coordinate = (row - rate, col + i)
                        pixel_magnitude = pixel_bilinear_coordinate(magnitude, pixel_coordinate)
                        if magnitude[row, col] == max(pixel_magnitude, magnitude[row, col]):
                            largest_magnitude[row, col] = magnitude[row, col]
                        else:
                            largest_magnitude[row, col] = 0
                            break

            elif -90 <= degree < -45:
                for i in np.arange(-2, 2.5, step):
                    if i == 0:
                        continue
                    else:
                        rate = -np.tan(np.deg2rad(90 - degree)) * i
                        pixel_coordinate = (row - i, col + rate)
                        pixel_magnitude = pixel_bilinear_coordinate(magnitude, pixel_coordinate)
                        if magnitude[row, col] == max(pixel_magnitude, magnitude[row, col]):
                            largest_magnitude[row, col] = magnitude[row, col]
                        else:
                            largest_magnitude[row, col] = 0
                            break

            else:
                print(row, col, 'error!  degree :', degree)

    return largest_magnitude

def double_thresholding(src, high_threshold):

    ####################################################################################
    # TODO
    # TODO double_thresholding 완성
    # TODO Goal : Weak Edge와 Strong Edge를 토대로 연결성을 찾아서 최종적인 Canny Edge Detection 이미지를 도출
    # TODO 이 함수는 건드릴 필요가 없음.
    # TODO largest_magnitude     : non_maximum_supression 결과(가장 강한 edge만 남김)
    # TODO double_thresholding 수행 high threshold value는 메인문에서 지정한 값만 사용할 것
    # TODO 3 x 3 non_maximum_supression의 high threshold 값: 40
    # TODO 5 x 5 non_maximum_supression의 high threshold 값: 29
    ####################################################################################

    dst = src.copy()
    # dst => 0 ~ 255
    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)
    (h, w) = dst.shape

    high_threshold_value = high_threshold

    low_threshold_value = high_threshold_value * 0.4

    for row in range(h):
        for col in range(w):
            if dst[row, col] >= high_threshold_value:
                dst[row, col] = 255
            elif dst[row, col] < low_threshold_value:
                dst[row, col] = 0
            else:
                ####################################################################
                # TODO
                # TODO Weak Edge일때 구현
                # TODO search_weak_edge 함수 설명 : Weak Edge를 찾아 배열에 저장하는 함수
                # TODO classify_edge : search_weak_edge를 통해 찾아낸 Weak Edge들을 이용하여 주변에
                #  Strong Edge가 있으면 weak Edge들을 Strong으로 변경 Edge 주변에 Strong이 없으면 Weak Edge를 버림.
                ####################################################################
                weak_edge = []
                weak_edge.append((row, col))
                # search_weak_edge(dst, weak_edge, high_threshold_value, low_threshold_value)
                search_weak_edge(dst, weak_edge, high_threshold_value, low_threshold_value, row, col)
                if classify_edge(dst, weak_edge, high_threshold_value):
                    for idx in range(len(weak_edge)):
                        (r, c) = weak_edge[idx]
                        dst[r, c] = 255
                else:
                    for idx in range(len(weak_edge)):
                        (r, c) = weak_edge[idx]
                        dst[r, c] = 0

    return dst

# 재귀로 만들기
def search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row, col):
    ####################################################################################
    # TODO
    # TODO search_weak_edge 함수
    # TODO Goal : 연결된 Weak Edge를 찾아서 저장하는 함수
    # TODO 구현의 자유도를 주기위해 실습을 참고하여 구현해도 되며
    # TODO 직접 생각해서 구현해도 무방함.
    ####################################################################################

    if ((row, col) in edges) or dst[row, col] < low_threshold_value or dst[row, col] > high_threshold_value:
        return edges

    edges.append((row, col))
    h, w = dst.shape

    # 1번 좌표
    if row > 0 and col > 0:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row - 1, col - 1)
    # 2번 좌표
    if row > 0:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row - 1, col)
    # 3번 좌표
    if row > 0 and col < w - 1:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row - 1, col + 1)
    # 4번 좌표
    if col > 0:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row, col - 1)
    # 6번 좌표
    if col < w - 1:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row, col + 1)
    # 7번 좌표
    if row < h - 1 and col > 0:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row + 1, col - 1)
    # 8번 좌표
    if row < h - 1:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row + 1, col)
    # 9번 좌표
    if row < h - 1 and col < w - 1:
        edges = search_weak_edge(dst, edges, high_threshold_value, low_threshold_value, row + 1, col + 1)

    return list(set(edges))

def classify_edge(dst, weak_edge, high_threshold_value):
    ####################################################################################
    # TODO
    # TODO weak edge가 strong edge랑 연결되어 있는지 확인한 후 edge임을 결정하는 함수
    # TODO 구현의 자유도를 주기위해 실습을 참고하여 구현해도 되며
    # TODO 직접 생각해서 구현해도 무방함.
    # strong edge 는 Th 보다 큰 것
    # weak_edge = search_weak_edge 의 반환 값
    ####################################################################################

    connected = False
    for pair in weak_edge:
        row = pair[0]
        col = pair[1]

        if dst[row - 1, col - 1] > high_threshold_value:
            connected = True
        if dst[row - 1, col] > high_threshold_value:
            connected = True
        if dst[row - 1, col + 1] > high_threshold_value:
            connected = True
        if dst[row, col - 1] > high_threshold_value:
            connected = True
        if dst[row, col + 1] > high_threshold_value:
            connected = True
        if dst[row + 1, col - 1] > high_threshold_value:
            connected = True
        if dst[row + 1, col] > high_threshold_value:
            connected = True
        if dst[row + 1, col + 1] > high_threshold_value:
            connected = True

    return connected

def my_canny_edge_detection(src, fsize=3, sigma=1):

    # low-pass filter를 이용하여 blur효과
    # high-pass filter를 이용하여 edge 검출
    # gaussian filter -> sobel filter 를 이용해서 2번 filtering
    # DoG_x, DoG_y = get_DoG_filter(fsize, sigma)
    DoG_x, DoG_y = get_DoG_filter()
    Ix = my_filtering(src, DoG_x)
    Iy = my_filtering(src, DoG_y)

    # magnitude와 angle을 구함
    magnitude = calcMagnitude(Ix, Iy)
    cv2.imshow('magnitude', convert_uint8(magnitude))

    angle = calcAngle(Ix, Iy)

    #non-maximum suppression 수행
    larger_magnitude2 = non_maximum_supression_three_size(magnitude, angle)
    cv2.imshow('NMS_Three', convert_uint8(larger_magnitude2))
    larger_magnitude3 = non_maximum_supression_five_size(magnitude, angle)
    cv2.imshow('NMS_Five', convert_uint8(larger_magnitude3))

    #double thresholding 수행
    dst = double_thresholding(larger_magnitude2,40)
    dst2 = double_thresholding(larger_magnitude3,29)
    return dst, dst2

def main():
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    dst, dst2 = my_canny_edge_detection(src)

    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.imshow('my canny edge detection2', dst2)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()


