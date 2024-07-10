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

def insert_bev2map(img, map, y_min = 50, y_max = 250, x_length = 84):
    y_padding = (map.shape[1] / 2) - y_max
    x_padding = (map.shape[0] / 2) - (x_length / 2)

    color = [1, 1, 1]  # 패딩 색상 (흰색)
    new_img = cv2.copyMakeBorder(img, y_padding, map.shape[1] - img.shape[1] - y_padding, x_padding, x_padding, cv2.BORDER_CONSTANT, value=color)
    map += new_img

    return map

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


