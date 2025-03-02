import os
import json
import struct
import cv2
import pandas as pd

class DataReader:
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def read_saved_data(self):
        """
        读取保存好的IMU数据，根据index.json文件来获取每个数据块的信息并逐块读取。
        """
        index_path = os.path.join(self.save_dir, "index.json")
        
        # 读取索引文件，获取每个数据块的信息
        try:
            with open(index_path, "r") as f:
                index = json.load(f)
        except FileNotFoundError:
            print(f"Error: The index file at {index_path} was not found.")
            return
        
        # 遍历每个数据块，逐块读取数据
        for chunk_info in index.get("chunks", []):
            chunk_filename = chunk_info["file"]
            chunk_path = os.path.join(self.save_dir, chunk_filename)
            self._read_chunk(chunk_path)

    def _read_chunk(self, chunk_filename):
        """
        读取一个数据块文件，解码并输出其中保存的IMU数据。
        """
        try:
            with open(chunk_filename, "rb") as f:
                # 读取块头：时间戳（纳秒）
                timestamp_ns = struct.unpack("<Q", f.read(8))[0]
                print(f"Reading chunk: {chunk_filename}, Timestamp: {timestamp_ns}")
                
                # 逐个读取数据点，结构为 <Q3f3f3f (时间戳, 加速度, 角速度, 磁场)
                while chunk_data := f.read(40):  # 每个数据点占40字节
                    timestamp, accel_x, accel_y, accel_z, gyro_x, gyro_y, gyro_z, mag_x, mag_y, mag_z = struct.unpack("<Q3f3f3f", chunk_data)
                    
                    # 打印数据或根据需求处理数据
                    print(f"Timestamp: {timestamp}, Accel: ({accel_x}, {accel_y}, {accel_z}), Gyro: ({gyro_x}, {gyro_y}, {gyro_z}), Mag: ({mag_x}, {mag_y}, {mag_z})")
        
        except FileNotFoundError:
            print(f"Error: Chunk file {chunk_filename} not found.")
            return

    def read_video(self, video_filename):
        """
        读取视频文件并返回帧数据。
        """
        video_path = os.path.join(self.save_dir, video_filename)
        if not os.path.exists(video_path):
            print(f"Error: Video file {video_path} not found.")
            return
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def read_imu_data(self, imu_filename):
        """
        读取IMU数据文件并返回DataFrame。
        """
        imu_path = os.path.join(self.save_dir, imu_filename)
        if not os.path.exists(imu_path):
            print(f"Error: IMU file {imu_path} not found.")
            return
        
        return pd.read_csv(imu_path)