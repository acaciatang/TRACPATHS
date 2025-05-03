"""cut videos, 10000 frames each."""

import cv2, glob

for vid in glob.glob("../Data/video/*.MP4"):
    print(vid)
    chunk=0
    f = 0

    cap = cv2.VideoCapture(vid)
    if not cap.isOpened():
        cap.open()
    outname=vid.replace('.MP4', '_'+str(chunk)+'.MP4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(outname, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    
    while cap.isOpened():
        if f < 1000:
            ret, img = cap.read()
            if not ret:
                break    
            out.write(img)
            f=f+1
            print(f)
        else:
            out.release()
            print('Wrote: '+outname)
            chunk=chunk+1
            f=0
            outname=vid.replace('.MP4', '_'+str(chunk)+'.MP4')
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(outname, fourcc, 30.0, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    print('Wrote: '+vid)
            
print('All done!')


            



