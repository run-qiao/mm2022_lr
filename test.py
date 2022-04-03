
import cv2 as cv

def video_demo():
    """
    打开视频
    """
    capture = cv.VideoCapture(r"E:\Video\test.MP4") #修改为自己路径
    flag = 1
    index=0
    internal=25
    while True:
        ret, frame = capture.read()
        if flag % internal == 0 and ret is True:
            # cv.imshow("frame", frame)
            cv.imwrite(r"E:\Video\test_frame\{:0>5d}.jpg".format(index), frame)
            index+=1
        flag += 1
        c = cv.waitKey(50)
        if c == 27:
            break


cv.namedWindow("frame", 0)
video_demo()
cv.destroyAllWindows()
