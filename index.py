import cv2
import time
from ultralytics import YOLO
import pygame
import numpy as np
import os

def show_detection_window():
    # YOLO 모델 초기화
    print("YOLO 모델 로딩 중...")
    model = YOLO('yolov8n.pt')
    
    # 카메라 초기화
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    # 카메라가 제대로 열렸는지 확인
    if not cap1.isOpened() or not cap2.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    
    print("카메라 초기화 성공")
    
    # Pygame 초기화
    pygame.init()
    
    # 화면 설정 (전체화면)
    info = pygame.display.Info()
    width, height = info.current_w, info.current_h  # 현재 모니터의 해상도 가져오기
    screen = pygame.display.set_mode((width, height), pygame.FULLSCREEN)
    pygame.display.set_caption("사람 감지 모니터")
    
    # 폰트 크기를 화면 크기에 맞게 조정
    font_size = int(height / 12)  # 화면 높이의 1/12 크기로 설정
    
    # 폰트 설정 (나눔고딕 또는 기본 폰트)
    try:
        # 맥OS 기본 한글 폰트
        if os.path.exists('/System/Library/Fonts/AppleSDGothicNeo.ttc'):
            font = pygame.font.Font('/System/Library/Fonts/AppleSDGothicNeo.ttc', font_size)
        # 윈도우 기본 한글 폰트
        elif os.path.exists('C:/Windows/Fonts/malgun.ttf'):
            font = pygame.font.Font('C:/Windows/Fonts/malgun.ttf', font_size)
        else:
            font = pygame.font.Font(None, font_size)  # 폰트를 찾지 못한 경우 기본 폰트
    except:
        font = pygame.font.Font(None, font_size)
    
    running = True
    clock = pygame.time.Clock()
    
    def check_person_detection(results, confidence_threshold=0.3):
        """사람 감지 여부와 최대 신뢰도를 반환"""
        if len(results[0].boxes) > 0:
            # 사람 클래스(0)에 대한 신뢰도 값들을 가져옴
            confidences = [float(conf) for conf in results[0].boxes.conf]
            max_confidence = max(confidences) if confidences else 0
            return max_confidence >= confidence_threshold, max_confidence
        return False, 0.0
    
    try:
        while running:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
            
            # 카메라에서 프레임 읽기
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            
            if ret1 and ret2:
                # YOLO로 각 프레임 분석 (confidence threshold 0.3으로 설정)
                results1 = model(frame1, classes=0)
                results2 = model(frame2, classes=0)
                
                # 사람 감지 여부와 신뢰도 확인
                person_detected1, conf1 = check_person_detection(results1)
                person_detected2, conf2 = check_person_detection(results2)
                
                # 배경색 설정
                if person_detected1 or person_detected2:
                    background_color = (255, 200, 200)  # 연한 빨간색
                else:
                    background_color = (200, 255, 200)  # 연한 초록색
                
                # 화면 그리기
                screen.fill(background_color)
                
                # 텍스트 렌더링 (신뢰도 포함)
                text1 = font.render(f"왼쪽: {'감지됨 ({:.1%})' if person_detected1 else '감지 안됨'}"
                                  .format(conf1), True, (0, 0, 0))
                text2 = font.render(f"오른쪽: {'감지됨 ({:.1%})' if person_detected2 else '감지 안됨'}"
                                  .format(conf2), True, (0, 0, 0))
                
                # 텍스트 위치 설정
                text1_rect = text1.get_rect(center=(width/2, height/3))
                text2_rect = text2.get_rect(center=(width/2, height*2/3))
                
                # 텍스트 그리기
                screen.blit(text1, text1_rect)
                screen.blit(text2, text2_rect)
                
                # 화면 업데이트
                pygame.display.flip()
            
            # FPS 제한
            clock.tick(30)
                
    finally:
        # 자원 해제
        cap1.release()
        cap2.release()
        pygame.quit()
        print("\n프로그램 종료")

if __name__ == "__main__":
    show_detection_window()
