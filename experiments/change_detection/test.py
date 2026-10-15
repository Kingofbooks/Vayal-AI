import cv2
import numpy as np
def main():
    path1=r"F:\ComputerVision_From_Scratch\vayalai\data\samples\before.jpg"
    path2=r"F:\ComputerVision_From_Scratch\vayalai\data\samples\after.jpg"
    img1=cv2.imread(path1)
    img2=cv2.imread(path2)
    if img1 is None or img2 is None:
        print("Could not load one or both images")
        return
    print(img1.shape)
    print(img2.shape)
    if img2.shape == img2.shape:
        gray_before=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray_after=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        
        print(gray_after.dtype)
        print(np.int16(gray_after).dtype)
        
        np.int16(gray_after)
        np.int16(gray_before)
        diff = cv2.absdiff(gray_before, gray_after)
        mid = np.median(diff)
        diff_abs = diff-mid
        print(diff.dtype)
        print(mid.dtype)
        print(diff_abs.dtype)

main()