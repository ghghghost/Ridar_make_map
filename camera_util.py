import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar, RPLidarException
from sklearn.cluster import DBSCAN
import time
import math
import cv2

max_resist = 154 #23.5
min_resist = 108 #-23.5
distance_per_sec = 760 #(mm/s) 
below_distance = 765 #(mm) from the wheel
car_width = 540 #(mm) real size
real_height = int(1235 / 2)
real_width = int(940 / 2)


def rotate_points(points, angle):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])
    rotated_points = points[:, :2] @ rotation_matrix.T
    if points.shape[1] > 2:
        rotated_points = np.hstack((rotated_points, points[:, 2:]))
    return rotated_points

def calculate_new_map(map, car_speed, car_angle, car_length, time_spent):
    drive_angle = math.pi * (car_angle / 180)
    alpha = 1
    beta = 1
    if car_angle < -0.1:
        alpha = -1
        car_angle = -1 * car_angle
    if car_speed < -0.1:
        beta = -1
        car_speed = -1 * car_speed

    if car_angle <= 0.1 and car_angle >= -0.1:
        map[:, 1] -= car_speed
    else:
        r = (car_length / math.tan(drive_angle / 2))
        theta = car_speed * time_spent / r
        map[:, 0] -= alpha * r * (1 - math.cos(theta))
        map[:, 1] -= beta * r * math.sin(theta)
        map = rotate_points(map, theta)

    return map

def transform_bev(image, src_points, dst_points):
        # 변환 행렬 계산
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # 결과 이미지 크기 (예: 원본 이미지와 동일한 크기)
    h, w = image.shape[:2]
    birds_eye_view = cv2.warpPerspective(image, M, (w, h))
    
    return birds_eye_view

def get_nonzero_coordinates(image):
    # 0이 아닌 값들의 좌표를 찾기
    nonzero_coords = np.transpose(np.nonzero(image))
    return nonzero_coords

def insert_bev2map(img, map, y_min = 50, y_max = 250, x_length = 84):
    y_padding = (map.shape[1] / 2) - y_max
    x_padding = (map.shape[0] / 2) - (x_length / 2)

    color = [1, 1, 1]  # 패딩 색상 (흰색)
    new_img = cv2.copyMakeBorder(img, y_padding, map.shape[1] - img.shape[1] - y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=color)
    map += new_img

    return map

# globally define
def get_centermap():
    # 2000을 20등분한 구간들의 중심점을 담는 리스트 생성
    start = 0
    end = 2000
    num_segments = 20

    # 각 구간의 길이
    segment_length = (end - start) / num_segments

    # 구간 중심점 리스트 초기화
    centers = []

    for i in range(num_segments):
        # 각 구간의 시작점과 끝점
        segment_start = start + i * segment_length
        segment_end = segment_start + segment_length
        # 중심점 계산
        center = (segment_start + segment_end) / 2
        centers.append(center)
    
    return centers

center_map = get_centermap()
window_height = 50
window_width = 100

def make_sw(img, prev_sw, opposite_sw, time_):
    now_sw = prev_sw.copy()
    prev_opposite = opposite_sw.copy()

    moving_distance = (time_ * 760) // 100  # 이동한 sliding window 개수

    if moving_distance < 20:
        prev_sw[:20-moving_distance] = now_sw[moving_distance:]
        prev_opposite[:20-moving_distance] = opposite_sw[moving_distance:]
        if 20 - moving_distance - 1 >= 0:
            fill_value = prev_sw[20 - moving_distance - 1]
            prev_sw[20 - moving_distance:] = [fill_value] * moving_distance

            fill_value2 = prev_opposite[20 - moving_distance - 1]
            prev_opposite[20 - moving_distance:] = [fill_value2] * moving_distance

    # 확실한 점들 먼저 sliding window
    for index, h in enumerate(now_sw):
        # 박스 영역 자르기
        box_region = img[h[1]-window_height:h[1]+window_height, h[0]-window_width:h[0]+window_width]

        # 0이 아닌 점들의 x 좌표 찾기
        non_zero_points = np.column_stack(np.where(box_region != 0))

        # 0이 아닌 점이 없는 경우
        if non_zero_points.size == 0:
            now_sw[index] = [0, h[1]]
            continue

        # x 좌표 평균 계산
        mean_x = np.mean(non_zero_points[:, 1])

        # 전체 이미지에서의 x 좌표로 변환
        mean_x_global = h[0] + mean_x - window_width

        now_sw[index] = [mean_x_global, h[1]]

    for index, h in enumerate(opposite_sw):
        # 박스 영역 자르기
        box_region = img[h[1]-window_height:h[1]+window_height, h[0]-window_width:h[0]+window_width]

        # 0이 아닌 점들의 x 좌표 찾기
        non_zero_points = np.column_stack(np.where(box_region != 0))

        # 0이 아닌 점이 없는 경우
        if non_zero_points.size == 0:
            opposite_sw[index] = [0, h[1]]
            continue

        # x 좌표 평균 계산
        mean_x = np.mean(non_zero_points[:, 1])

        # 전체 이미지에서의 x 좌표로 변환
        mean_x_global = h[0] + mean_x - window_width

        opposite_sw[index] = [mean_x_global, h[1]]

    # 0인 점: 위아래(3칸까지 확인) 기준으로 평균내기
    for index, h in enumerate(now_sw):
        if h[0] != 0:
            continue

        flag_up = False
        flag_down = False

        tmp_up = 0
        tmp_down = 0

        for i in range(3):
            if flag_up and flag_down:
                break

            if not flag_up and index + i + 1 < len(now_sw) and now_sw[index + i + 1][0] != 0:
                flag_up = True
                tmp_up = now_sw[index + i + 1][0]

            if not flag_down and index - i - 1 >= 0 and now_sw[index - i - 1][0] != 0:
                flag_down = True
                tmp_down = now_sw[index - i - 1][0]

        # 한쪽 없으면 보이는 쪽과 똑같은 값
        if tmp_down + tmp_up != 0:
            if tmp_up == 0:
                now_sw[index] = [tmp_down, h[1]]
            elif tmp_down == 0:
                now_sw[index] = [tmp_up, h[1]]
            else:
                now_sw[index] = [(tmp_up + tmp_down) / 2, h[1]]

            # 3칸 이내에 있으면 평균 낸 값과 반대쪽 값의 평균 값 내기
            if opposite_sw[index][0] != 0:
                offset = 840 if opposite_sw[index][0] > 1000 else -840
                now_sw[index][0] = (now_sw[index][0] + (opposite_sw[index][0] + offset)) / 2
        else:
            if opposite_sw[index][0] == 0:
                now_sw[index][0] = prev_sw[index][0]
            else:
                offset = 840 if opposite_sw[index][0] > 1000 else -840
                now_sw[index][0] = opposite_sw[index][0] + offset

    for index, h in enumerate(opposite_sw):
        if h[0] != 0:
            continue

        flag_up = False
        flag_down = False

        tmp_up = 0
        tmp_down = 0

        for i in range(3):
            if flag_up and flag_down:
                break

            if not flag_up and index + i + 1 < len(opposite_sw) and opposite_sw[index + i + 1][0] != 0:
                flag_up = True
                tmp_up = opposite_sw[index + i + 1][0]

            if not flag_down and index - i - 1 >= 0 and opposite_sw[index - i - 1][0] != 0:
                flag_down = True
                tmp_down = opposite_sw[index - i - 1][0]

        # 한쪽 없으면 보이는 쪽과 똑같은 값
        if tmp_down + tmp_up != 0:
            if tmp_up == 0:
                opposite_sw[index] = [tmp_down, h[1]]
            elif tmp_down == 0:
                opposite_sw[index] = [tmp_up, h[1]]
            else:
                opposite_sw[index] = [(tmp_up + tmp_down) / 2, h[1]]

            # 3칸 이내에 있으면 평균 낸 값과 반대쪽 값의 평균 값 내기
            if now_sw[index][0] != 0:
                offset = 840 if now_sw[index][0] > 1000 else -840
                opposite_sw[index][0] = (opposite_sw[index][0] + (now_sw[index][0] + offset)) / 2
        else:
            if now_sw[index][0] == 0:
                opposite_sw[index][0] = prev_opposite[index][0]
            else:
                offset = 840 if now_sw[index][0] > 1000 else -840
                opposite_sw[index][0] = now_sw[index][0] + offset

    return now_sw, opposite_sw

def calculate_mean_x(data, start_index, end_index):
    # 시작 인덱스와 끝 인덱스 사이의 x 좌표 추출
    x_coords = [point[0] for point in data[start_index:end_index+1]]
    
    # x 좌표의 평균 계산
    mean_x = sum(x_coords) / len(x_coords)
    
    return mean_x

def get_angle_speed(left_sw, right_sw, left_class, box_num):
    speed = 254
    angle1 = 0
    angle2 = 0

    if left_class == 0:
        mean_x = calculate_mean_x(left_sw, 0, 4)
        distance_mid = 1000 - mean_x
        angle1 =  distance_mid - 300
    else:
        mean_x = calculate_mean_x(right_sw, 0, 4)
        distance_mid = 1000 - mean_x
        angle1 =  distance_mid - 300
    
    mid_ = (left_sw[box_num][0] + right_sw[box_num][0]) / 2
    angle2 = box_num * 100 / (mid_ - 1000)

    angle = angle1 * 0.2 + angle2 * 0.8
    return speed, angle




# left_max: -23.5 degree, right_max: 23.5 degree
# return: resist value
def convert_angle2resist(angle):
    if (angle > 23): angle = 23
    elif (angle < -23): angle = -23

    return ((max_resist + min_resist) // 2) + angle

# # 시각화 설정
# plt.ion()  # 인터랙티브 모드 켜기
# fig, ax = plt.subplots()
# ax.set_xlim(-2000, 2000)  # x축 범위 고정
# ax.set_ylim(-2000, 2000)  # y축 범위 고정
# ax.set_xlabel('X (m)')
# ax.set_ylabel('Y (m)')
# ax.set_title('Real-Time LiDAR Data Visualization')
# ax.grid(True)

# time_spent = 0
# car_speed = 760.0 # (mm/s)
# car_angle = 30  # suppose angle is 30 degrees
# car_length = 540.0  # suppose the car's length is 80 cm

# last_5_point_maps = []

# try:
#     # 라이다 스캔 시작
#     start_time = time.time()
#     for scan in lidar.iter_scans(max_buf_meas=5000):
#         x, y, z = process_lidar_data(scan)
#         points = np.vstack((-y, -x)).T  # 좌표 변환: 90도 회전 + 좌우반전

#         # DBSCAN 클러스터링 실행
#         dbscan = DBSCAN(eps=150, min_samples=5)
#         labels = dbscan.fit_predict(points[:, :2])

#         # 클러스터링 결과 시각화
#         ax.cla()
#         unique_labels = set(labels)
#         colors = plt.colormaps["tab20"](np.linspace(0, 1, len(unique_labels)))

#         for label in unique_labels:
#             if label == -1:
#                 continue  # Skip noise
#             class_member_mask = (labels == label)
#             xy = points[class_member_mask]
#             color = colors[label % len(unique_labels)]
#             ax.scatter(xy[:, 0], xy[:, 1], s=1, label=f'Cluster {label}', color=color)

#         ax.scatter(0, 0, color='red', s=50)  # 원점을 빨간색으로 표시
#         ax.set_xlim(-1500, 1500)  # x축 범위 고정
#         ax.set_ylim(-1500, 1500)  # y축 범위 고정

#         for i, map_points in enumerate(last_5_point_maps):
#             map_points = calculate_new_map(map_points, car_speed, car_angle, car_length, time_spent)
#             last_5_point_maps[i] = map_points
#             ax.scatter(map_points[:, 0], map_points[:, 1], s=1, label=f'Scan {i + 1}')

#         last_5_point_maps.append(points)
#         if len(last_5_point_maps) > 15:
#             last_5_point_maps.pop(0)
#             lidar.clean_input()

#         ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         # plt.pause(0.00001)

#         end_time = time.time()
#         time_spent = end_time - start_time
#         print(f"Time spent on 5 iterations: {time_spent} seconds")
#         start_time = end_time


