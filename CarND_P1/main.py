
import cv2
import os 
import argparse
import matplotlib.pyplot as plt
from pipeline import color_frame_pipeline
from os.path import join, basename
from collections import deque

ap = argparse.ArgumentParser()
ap.add_argument("--img", "--images", help="Relative path to test images folder")
ap.add_argument("--vid", "--videos", help="Relative path to test videos folder" )
ap.add_argument("--fps", help="FPS rate for videos")
ap.add_argument("--o", "--out_dir", help="Output path")
args = ap.parse_args()


if __name__ == "__main__":
  test_images_dir = join(args.img)
  test_images = [join(test_images_dir, name) for name in os.listdir(test_images_dir)]

  for test_img in test_images:
    print(f"[INFO] Processing {test_img}...")
    in_image = cv2.cvtColor(cv2.imread(test_img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    out_image = color_frame_pipeline([in_image], solid_lines=False)
    print(f"[INFO] Saving img")

    
  test_videos_dir = join(args.vid)
  test_videos = [join(test_videos_dir, name) for name in os.listdir(test_videos_dir)]
  
  resize_h, resize_w = 540, 960

  for test_video in test_videos:
    print(f"[INFO] processing {test_video}")

    cap = cv2.VideoCapture(test_video)
    out = cv2.VideoWriter(join(args.o, basename(test_video)),fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
                                fps=20.0, frameSize=(resize_h, resize_w))

    frame_buffer = deque(maxlen=10)

    while cap.isOpened():
      ret, color_frame = cap.read()
      if ret:
        color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        color_frame = cv2.resize(color_frame, (resize_h, resize_w))

        frame_buffer.append(color_frame)
        blend_frame = color_frame_pipeline(frames=frame_buffer, 
                                           solid_lines=True, 
                                           temporal_smoothing=True)
        out.write(cv2.cvtColor(blend_frame, cv2.COLOR_RGB2BGR))
      else:
        break

    cap.release()
    out.release()

    



