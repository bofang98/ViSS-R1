import cv2
import os

# 设置帧率
fps = 30

# 获取当前目录下的所有子文件夹
root_dir = '/root/paddlejob/workspace/env_run/gpubox03_ssd5/fangbo05/Video-R1/Evaluation/MVBench/tvqa/frames_fps3_hq'
folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

for folder in folders:
    # 获取文件夹中的所有图片
    images = sorted([img for img in os.listdir(os.path.join(root_dir, folder)) if img.endswith('.jpg')])
    
    # 读取第一张图片以获取视频尺寸
    first_image_path = os.path.join(root_dir, folder, images[0])
    frame = cv2.imread(first_image_path)
    height, width, _ = frame.shape

    # 创建视频写入对象
    video_filename = f"{folder}.mp4"
    video_path = os.path.join(root_dir, video_filename)
    video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    # 将每一张图片写入视频
    for image in images:
        image_path = os.path.join(root_dir, folder, image)
        frame = cv2.imread(image_path)
        video.write(frame)

    # 释放视频写入对象
    video.release()

    print(f"Video saved: {video_path}")
