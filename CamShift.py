import cv2

def camShift():
    global frame,frame2,inputmode,trackWindow,roi_hist

    try:
        # 저장된 영상 불러옴
        cap = cv2.VideoCapture(0)
        cap.set(3,480)
        cap.set(4,320)
    except Exception as e:
        print(e)
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame',onMouse,param=(frame,frame2))

    # meanShift 함수의 3번째 인자. 10회 반복 혹은 C1_o ~ C1_r의 차이가 1pt 날 때까지 작동
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10,1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if trackWindow is not None:
            hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
            ret, trackWindow = cv2.CamShift(dst,trackWindow,termination)

            ### 이 부분 수정
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            cv2.polylines(frame,[pts],True,(0,255,0),2)

        cv2.imshow('frame',frame)

        k = cv2.waitKey(60) & 0xFF
        if k == 27:
            break
        # i를 눌러서 영상을 멈춰서 roi 설정    
        if k == ord('i'):
            print("Meanshift를 위한 지역을 선택하고 키를 입력해라")
            inputmode = True
            frame2 = frame.copy()

            while inputmode:
                cv2.imshow('frame',frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

camShift()