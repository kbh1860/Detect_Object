import cv2 as cv

cap = cv.VideoCapture(0)
# If You Want Capture The Video -> Make ViceoCapture()
#VideoCapture(Index)
# If Camera is Built In Your Computer Index is 0
# If Comera is External And Index is 1


if cap.isOpened() == False: #Can't Open The Camera
    print("Failed To Open Camera.")
    exit(1)

while(True):
    Ret , Image_Frame = cap.read()  #Read The Image From Camera

    if(Ret == False): # Can't Read The Image From Camera
        print("Failed To Capture The Camera")
        break
      
    cv.imshow('Color', Image_Frame) #Show The Read Image

    key = cv.waitKey(1)

    if(key == 27): #If Key is ESC And Break The Loop
        break

cap.release() #Close The Accss Camera 
cv.destroyAllWindows()
