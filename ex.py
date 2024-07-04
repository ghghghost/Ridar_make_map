import numpy as np
import matplotlib.pyplot as plt
from rplidar import RPLidar, RPLidarException
from sklearn.cluster import DBSCAN
import time
import math

# RPLidar 설정
PORT_NAME = '/dev/ttyUSB0'  # 라이다가 연결된 포트 이름
lidar = RPLidar(port= PORT_NAME, timeout=3)
print(lidar)

def process_lidar_data(scan):
    # 스캔 데이터에서 각도와 거리를 추출
    angles = np.array([measurement[1] for measurement in scan])
    distances = np.array([measurement[2] for measurement in scan])

    # 각도와 거리를 x, y, z 좌표로 변환
    x = distances * np.cos(np.deg2rad(angles))
    y = distances * np.sin(np.deg2rad(angles))
    z = np.zeros_like(x)  # 2D 라이다이므로 z는 0으로 설정

    return x, y, z

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

# 시각화 설정
plt.ion()  # 인터랙티브 모드 켜기
fig, ax = plt.subplots()
ax.set_xlim(-2000, 2000)  # x축 범위 고정
ax.set_ylim(-2000, 2000)  # y축 범위 고정
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_title('Real-Time LiDAR Data Visualization')
ax.grid(True)

time_spent = 0
car_speed = 500.0
car_angle = 30  # suppose angle is 30 degrees
car_length = 800.0  # suppose the car's length is 80 cm

last_5_point_maps = []

try:
    # 라이다 스캔 시작
    start_time = time.time()
    for scan in lidar.iter_scans(max_buf_meas=5000):
        x, y, z = process_lidar_data(scan)
        points = np.vstack((-y, -x)).T  # 좌표 변환: 90도 회전 + 좌우반전

        # DBSCAN 클러스터링 실행
        dbscan = DBSCAN(eps=150, min_samples=5)
        labels = dbscan.fit_predict(points[:, :2])

        # 클러스터링 결과 시각화
        ax.cla()
        unique_labels = set(labels)
        colors = plt.colormaps["tab20"](np.linspace(0, 1, len(unique_labels)))

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise
            class_member_mask = (labels == label)
            xy = points[class_member_mask]
            color = colors[label % len(unique_labels)]
            ax.scatter(xy[:, 0], xy[:, 1], s=1, label=f'Cluster {label}', color=color)

        ax.scatter(0, 0, color='red', s=50)  # 원점을 빨간색으로 표시
        ax.set_xlim(-1500, 1500)  # x축 범위 고정
        ax.set_ylim(-1500, 1500)  # y축 범위 고정

        for i, map_points in enumerate(last_5_point_maps):
            map_points = calculate_new_map(map_points, car_speed, car_angle, car_length, time_spent)
            last_5_point_maps[i] = map_points
            ax.scatter(map_points[:, 0], map_points[:, 1], s=1, label=f'Scan {i + 1}')

        last_5_point_maps.append(points)
        if len(last_5_point_maps) > 15:
            last_5_point_maps.pop(0)
            lidar.clean_input()

        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        # plt.pause(0.00001)

        end_time = time.time()
        time_spent = end_time - start_time
        print(f"Time spent on 5 iterations: {time_spent} seconds")
        start_time = end_time


except KeyboardInterrupt:
    print('Stopping.')
except RPLidarException as e:
    print(f'RPLidar exception: {e}')

lidar.stop()
lidar.disconnect()
plt.ioff()
plt.show()
