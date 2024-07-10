
import cv2
import time
import keyboard

def save_frames_from_camera(interval=0.5):
    # 카메라 객체 생성 (기본 카메라 장치 사용)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라가 준비되었습니다. 'Enter' 키를 눌러 시작하세요.")

    # 'Enter' 키가 눌릴 때까지 대기
    keyboard.wait('enter')
    print("'Enter' 키가 눌렸습니다. 이미지 저장을 시작합니다.")

    frame_count = 0
    start_time = time.time()

    while True:
        # 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        current_time = time.time()
        elapsed_time = current_time - start_time

        # 0.5초 간격으로 프레임 저장
        if elapsed_time >= interval:
            frame_count += 1
            filename = f"." \
                       f"/interval/frame_{frame_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"이미지 저장됨: {filename}")
            start_time = current_time

        # ESC 키를 누르면 종료
        if keyboard.is_pressed('esc'):
            print("ESC 키가 눌렸습니다. 프로그램을 종료합니다.")
            break

        # 프레임을 윈도우에 디스플레이
        cv2.imshow('Camera', frame)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

# 함수 호출
save_frames_from_camera()
