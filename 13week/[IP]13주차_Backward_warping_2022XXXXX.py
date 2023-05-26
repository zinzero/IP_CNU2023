import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_roi_coordinates(src):

    """
    :param src: 변환 시킬 이미지
    :return: 변환 시킬 이미지에서의 ROI 좌표
    """

    #####################################################
    # TODO
    # TODO Warping한 이미지의 RoI 좌표를 추출하는 방식
    # TODO 보정한 이미지의 4개의 꼭짓점(이미지의 틀)이 아님.
    # TODO ROI 좌표가 4개 이상으로 나와도 상관 없음.
    # TODO 실습 PPT 참고
    #####################################################

    h, w = src.shape

    coordinates_list = []

    # 4개의 좌표값을 도출해야 함.

    return coordinates_list

def get_max_min_coordinates(roi_coordinates, M):

    """
    :param roi_coordinates: roi 좌표
    :param M: 변환 행렬
    :return: 변환된 좌표 값들의 최댓값 및 최솟값
    """

    #####################################################
    # TODO
    # TODO Warping한 이미지를 보정하는 과정
    # TODO 매개변수로 RoI 좌표를 기반으로 전체적인 이미지 크기를 추출
    # TODO M은 3 X 3 크기의 행렬
    # TODO 실습 PPT 참고
    #####################################################

    # dst shape 구하기
    cor_transform = []
    # Original에서 M에 의해 변환된 좌표의 최대 최소 범위 파악

    cor_transform = np.array(cor_transform)

    # 추출한 좌표들을 기반으로 행의 최소, 최대값 열의 최소 최댓값을 추출
    row_max = ???
    row_min = ???
    col_max = ???
    col_min = ???

    return row_max, row_min, col_max, col_min

def backward(src, M):

    """
    :param src: 변환 시킬 이미지
    :param M: 변환 행렬
    :return: dst: 크기가 보정된 변환 이미지
    """

    ##############################################################
    # TODO
    # TODO backward 완성
    # TODO 구현 사항
    # TODO 1. ROI 좌표를 구하는 함수 구현 (get_roi_coordinates)
    # TODO 2. 변환된 이미지의 크기 구하기
    #   TODO 2.1 ROI 좌표들의 각 축 방향별로 최솟값 및 최댓값 구하기
    #        -> get_max_min_coordinates 함수 구현
    #   TODO 2.2 결과 이미지의 크기 구하기
    # TODO 3. Backward warping 구현
    ##############################################################

    print('< backward >')
    print('M')
    print(M)

    h, w = src.shape
    # M 역행렬 구하기
    M_inv = np.linalg.inv(M)
    print('M inv')
    print(M_inv)

    # 중복 제거
    roi_coordinates = list(set(get_roi_coordinates(src)))


    (row_max, row_min, col_max, col_min) = get_max_min_coordinates(roi_coordinates, M)


    ############################################
    # TODO
    # TODO 2.2 결과 이미지의 크기 구하기
    ############################################

    h_ = ???
    w_ = ???
    dst = np.zeros((h_, w_))

    for row in range(h_):
        for col in range(w_):

            #################################################################
            # TODO
            # TODO 3. Backward warping 구현
            # TODO 3. 세부 사항
            # TODO  P_dst: 변환시키기 전의 이미지 좌표에서 변환 행렬 M에 의해 실제 변환된 좌표
            # TODO NOTE. 현재 dst의 좌표는 모두 정수 즉 보정된 좌표계이다
            # TODO  src_col: 역변환(M의 inverse)에 의한 x좌표
            # TODO  src_row: 역변환(M의 inverse)에 의한 y좌표
            ##################################################################

            P_dst = ???

            # original 좌표로 매핑
            P = np.dot(M_inv, P_dst)
            src_col = ???
            src_row = ???
            # bilinear interpolation

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            # index를 초과하는 부분에 대해서는 값을 채우지 않음
            if src_col_right >= w or src_row_bottom >= h or src_col_left < 0 or src_row_top < 0:
                continue

            #################################################################
            # TODO
            # TODO 3. Backward warping 구현
            # TODO Bilinear interpolation 완성
            ##################################################################

            intensity = ???

            dst[row, col] = intensity

    dst = dst.astype(np.uint8)

    print('dst shape : {}'.format(dst.shape))
    print('dst min : {} dst max : {}'.format(np.min(dst),np.max(dst)))

    return dst

def generate_rotation(degree):
    M_ro = ???
    return M_ro

def generate_scaling(x_scaling, y_scaling):
    M_sc = ???
    return M_sc

def generate_shearing(x, y):
    M_sh = ???
    return M_sh

def display_image(image_list):
    template = np.zeros((850,800))
    template = cv2.line(template, (0, 25), (800, 25), (0.5, 0.5, 0.5))
    template = cv2.line(template, (0, 425), (800, 425), (0.5, 0.5, 0.5))
    template = cv2.line(template, (0, 450), (800, 450), (0.5, 0.5, 0.5))
    template = cv2.line(template, (400, 0), (400, 850), (0.5, 0.5, 0.5))

    template = cv2.putText(template, 'step 1 rotation', (100, 18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255,0,0), 1,
                           cv2.LINE_4)
    template = cv2.putText(template, 'step 2 shear', (525, 18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1,
                           cv2.LINE_4)
    template = cv2.putText(template, 'step 3 scale', (100, 425 + 18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1,
                           cv2.LINE_4)
    template = cv2.putText(template, 'final image', (525, 425 + 18), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 0), 1,
                           cv2.LINE_4)

    template[26:424, 1:399] = 255
    template[26:424, 401:799] = 255
    template[451:849, 1:399] = 255
    template[451:849, 401:799] = 255

    # first_image
    first_image = image_list[0] / 255
    first_image_shape = first_image.shape
    coord_1 = (400 - first_image_shape[0]) // 2
    coord_2 = (400 - first_image_shape[1]) // 2
    template[25 + coord_1 : 25 + first_image_shape[0] + coord_1,  coord_2 : first_image_shape[1] + coord_2] = first_image

    if len(image_list) == 2:
        second_image = image_list[1] / 255
        second_image_shape = second_image.shape
        coord_1 = (400 - second_image_shape[0]) // 2
        coord_2 = (400 - second_image_shape[1]) // 2
        coord_2 += 400
        template[25 + coord_1: 25 + second_image_shape[0] + coord_1,
        coord_2: second_image_shape[1] + coord_2] = second_image

    if len(image_list) == 3:
        second_image = image_list[1] / 255
        second_image_shape = second_image.shape
        coord_1 = (400 - second_image_shape[0]) // 2
        coord_2 = (400 - second_image_shape[1]) // 2
        coord_2 += 400
        template[25 + coord_1: 25 + second_image_shape[0] + coord_1,
        coord_2: second_image_shape[1] + coord_2] = second_image

        third_image = image_list[2] / 255
        third_image_shape = third_image.shape
        coord_1 = (400 - third_image_shape[0]) // 2
        coord_2 = (400 - third_image_shape[1]) // 2
        coord_1 += 425
        template[25 + coord_1: 25 + third_image_shape[0] + coord_1,
        coord_2: third_image_shape[1] + coord_2] = third_image

    if len(image_list) == 4:

        second_image = image_list[1] / 255
        second_image_shape = second_image.shape
        coord_1 = (400 - second_image_shape[0]) // 2
        coord_2 = (400 - second_image_shape[1]) // 2
        coord_2 += 400
        template[25 + coord_1: 25 + second_image_shape[0] + coord_1,
        coord_2: second_image_shape[1] + coord_2] = second_image

        third_image = image_list[2] / 255
        third_image_shape = third_image.shape
        coord_1 = (400 - third_image_shape[0]) // 2
        coord_2 = (400 - third_image_shape[1]) // 2
        coord_1 += 425

        template[25 + coord_1: 25 + third_image_shape[0] + coord_1,
        coord_2: third_image_shape[1] + coord_2] = third_image

        final_image = image_list[3] / 255
        final_image_shape = final_image.shape
        coord_1 = (400 - final_image_shape[0]) // 2
        coord_2 = (400 - final_image_shape[1]) // 2
        coord_1 += 425
        coord_2 += 400
        template[25 + coord_1: 25 + final_image_shape[0] + coord_1,
        coord_2: final_image_shape[1] + coord_2] = final_image

    cv2.imshow('image',template)
    cv2.waitKey()

def main():

    src = cv2.imread('./Lena.png', cv2.IMREAD_GRAYSCALE)
    src = cv2.resize(src, (285, 285))

    ###########################################################
    # TODO
    # TODO M 완성
    # TODO M_tr, M_sc ... 등등 모든 행렬 M 완성하기
    # TODO 각 변환 행렬 행성 시 generate_xxxx 함수의 인자값을 참고하여 구현
    ###########################################################

    degree = 20
    # scaling
    M_sc = generate_scaling(0.75, 0.75)
    # rotation
    M_ro = generate_rotation(degree)
    # shearing
    M_sh = generate_shearing(0.2, 0.2)

    # rotation -> Shear -> Scale
    M = M_ro
    M = np.dot(M_sh, M)
    M = np.dot(M_sc, M)

    # backward step by step
    dst_back1 = backward(src, M_ro)
    display_image([dst_back1])

    dst_back2 = backward(dst_back1, M_sh)
    display_image([dst_back1,dst_back2])

    dst_back3 = backward(dst_back2, M_sc)
    display_image([dst_back1, dst_back2, dst_back3])

    # backward all dot matrix
    dst_back4 = backward(src, M)
    display_image([dst_back1, dst_back2, dst_back3, dst_back4])

if __name__ =='__main__':
    main()