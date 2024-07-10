import cv2
import numpy as np
import os

# 체커보드의 차원 정의 (행과 열당 내부 코너 수)
CHECKERBOARD = (6, 9)

# 코너 검출 기준 정의
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# 3D 점의 세계 좌표 정의
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# 각 체커보드 이미지에 대한 3D 점 벡터 및 2D 점 벡터를 저장할 리스트 생성
objpoints = []  # 3D 점
imgpoints = []  # 2D 점

# 예시 이미지 파일 경로
image_path = 'new_auto/pictures2/frame_1.jpg'

# 이미지 읽기
img = cv2.imread(image_path)
if img is None:
    print(f"Could not read image {image_path}")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)
    
    if ret:
        # 코너 위치 미세 조정
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 3D 점과 2D 점 추가
        objpoints.append(objp)
        imgpoints.append(corners2)
        
        # 코너 그리기 및 표시
        img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        
        # 결과 이미지 저장 및 표시
        debug_image_path = os.path.join('./new_auto', 'debug_frame_17.jpg')
        cv2.imwrite(debug_image_path, img)
        
        cv2.imshow('img', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Chessboard corners not found in image {image_path}")
