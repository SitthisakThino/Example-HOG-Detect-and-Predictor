import dlib
import cv2

detector = dlib.fhog_object_detector('detector01.svm')
predictor = dlib.shape_predictor('shapePredict.dat')

cap = cv2.VideoCapture('SJCM0021.mp4')

while True:
    ret,frame = cap.read()
    if ret:
        frame = cv2.resize(frame,(1280,720))
        cv2.imshow('frame',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)

        for rect in rects:
            (x, y, w, h) = (rect.left(),rect.top(),(rect.right() - rect.left()),rect.bottom() - rect.top()) 
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            shape = predictor(gray, rect)
            for i in range(len(shape.parts())):
                cv2.circle(frame,(shape.part(i).x,shape.part(i).y),3,(255,255,255),-1)
        
        cv2.imshow('detect frame',frame)
    else:
        break

    key = cv2.waitKey(1)
    if key == ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()