import numpy as np
from collections import deque
import cv2
import os
class ChangeDetection:
    def __init__(self):
        print("Change Deduction has started!")

    def find_regions(self,mask):
        n,m=mask.shape
        l=[-1, 1, 0, 0] 
        r=[ 0, 0,-1, 1]
        labels=np.zeros(mask.shape, dtype=np.int32)
        ct=0
        st=deque()
        vis=np.zeros(mask.shape, dtype=bool)
        boundary = np.zeros(mask.shape, dtype=np.uint8)
        region = []
        for i in range(n):
            for j in range(m):
                val=mask[i][j]
                if val != 255 or vis[i][j]:
                    continue
                ct+=1
                region=[]
                st = deque([(i, j)])
                vis[i][j]=True
                labels[i,j]=ct
                while st:
                    x,y=st.pop()
                    region.append((x, y))
                    for k in range(4):
                        rr=x+l[k]
                        cc=y+r[k]
                        
                        if 0 <= rr < n and 0 <= cc < m and mask[rr][cc] == 255 and not vis[rr][cc]:
                            vis[rr][cc]=True
                            labels[rr,cc]=ct
                            st.append([rr,cc])
                for x, y in region:
                    for k in range(4):
                        rr = x + l[k]
                        cc = y + r[k]      
                        
                        if not (0 <= rr < n and 0 <= cc < m):
                            boundary[x, y] = 255
                            break 
                        if mask[rr, cc] == 0:
                            boundary[x, y] = 255
                            break     
        
        print("Region labels:")
        print(labels)

        print("\nBoundary mask:")
        print(boundary)
        
        print("Total count:")
        print(ct)
        
        return labels, boundary, ct
    
    def find_regions_fast(self, mask, image):
        count, labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                mask,
                connectivity=8
            )
        )

        for label in range(1, count):
            x, y, w, h, area = stats[label]

            # Ignore tiny regions for now
            if area < 20:
                continue

            cv2.rectangle(
                image,
                (x, y),
                (x + w - 1, y + h - 1),
                (0, 255, 0),
                2
            )

        print("Detected regions:", count - 1)
        cv2.imwrite(
            "data/processed/change_detection_result.jpg",
            image
        )

        return labels
    def rec_box(self,mask,label,image):
        for l in range(1,label+1):
            rows,cols=np.where(mask==l)
            x1, y1 = int(cols.min()), int(rows.min())
            x2, y2 = int(cols.max()), int(rows.max())
            print([x1,y1],[x2,y2])
            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )
        
        output_path = r"data\\processed\\change_detection_result.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Image successfully saved to {output_path}")
        else:
            print(f"Failed to save image to {output_path}")
            
        cv2.imshow("Drawing", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def make_brighter(self, image):
        brighter_img = cv2.convertScaleAbs(image, alpha=1.0, beta=50)
        output_path = r"data\\samples\\brighter_img_before.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        success = cv2.imwrite(output_path, brighter_img)
        if success:
            print(f"Image successfully saved to {output_path}")
        else:
            print(f"Failed to save image to {output_path}")
        return brighter_img
    
    def read_data(self,path1,path2):
        img1=cv2.imread(path1)
        img2=cv2.imread(path2)
        if img1 is None or img2 is None:
            print("Could not load one or both images")
            return
        print(img1.shape)
        print(img2.shape)
        new_img2=self.make_brighter(img2)
        gray_before=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
        gray_after=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
        before16 = gray_before.astype(np.int16)
        after16 = gray_after.astype(np.int16)
        
        signed_diff = after16 - before16
        brightness_shift = np.median(signed_diff)
        compensated = np.abs(signed_diff - brightness_shift)
        compensated = np.clip(compensated, 0, 255).astype(np.uint8)
        _, mask = cv2.threshold(
            compensated, 25, 255, cv2.THRESH_BINARY
        )
        # diff = cv2.absdiff(gray_before, gray_after)
        # mid = np.median(diff)
        # diff_abs = diff-mid
        # _, mask = cv2.threshold(
        #     diff_abs, 25, 255, cv2.THRESH_BINARY
        # )
        print("Estimated brightness shift:", brightness_shift)
        print("Changed pixels:", np.count_nonzero(mask))
        print("Mask dtype:", mask.dtype)
        print("Original mean:", img2.mean())
        print("Brighter mean:", new_img2.mean())
        print("Mean absolute difference:",cv2.absdiff(img2, new_img2).mean())
        
        
        return img1,img2,mask
                                
def main():
    change=ChangeDetection()
    mask0 = np.array([
        [0,   0,   0,   0, 0],
        [0, 255, 255,   0, 0],
        [0, 255, 255,   0, 0],
        [0,   0,   0, 255, 0],
        [0,   0,   0,   0, 0]
    ], dtype=np.uint8)
    mask1 = np.array([
            [255, 0, 255],
            [255, 0, 255],
            [255, 0, 255],
        ], dtype=np.uint8)
    mask2 = np.array([
                [255, 255, 255],
                [255, 255, 255],
                [255, 255, 255],
            ], dtype=np.uint8)
    path1=r"F:\ComputerVision_From_Scratch\vayalai\data\samples\before.jpg"
    path2=r"F:\ComputerVision_From_Scratch\vayalai\data\samples\after.jpg"
    img1,img2,mask =  change.read_data(path1,path2)
    # labels, boundary, ct = change.find_regions(mask)
    # change.rec_box(labels,ct,img2)
    change.find_regions_fast(mask, img2.copy())

if __name__=='__main__':
    main()