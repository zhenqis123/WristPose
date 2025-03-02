import socket
import cv2
import pickle
import struct
import os
import sys
import time
import json
import numpy as np
from PyQt5.QtGui import QPen, QPainter
from PyQt5.QtCore import QThread, pyqtSignal, Qt, pyqtSlot, QObject
from PyQt5.QtChart import QChart, QChartView, QLineSeries, QValueAxis
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, 
                            QPushButton, QLabel, QHBoxLayout, 
                            QStatusBar, QMessageBox, QFileDialog, QInputDialog, QLineEdit)
from PyQt5.QtGui import QImage, QPixmap, QColor
from PyQt5.QtCore import QTimer
from collections import deque
from turbojpeg import TurboJPEG, TJFLAG_FASTDCT
from threading import Event, Thread
from datetime import datetime
from io import BufferedWriter
from PyQt5.QtCore import QPointF
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import QSizePolicy
from threading import Lock
import threading
import multiprocessing
# from multiprocessing import Queue
from queue import Queue

# 创建一个信号类用于进程间通信
class Communicate(QObject):
    update_image_signal = pyqtSignal(str, str)  # 发送手势名称和路径

class MatplotlibCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, 
                                   QSizePolicy.Expanding, 
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
DATA_DIR = 'data/data'
os.makedirs(DATA_DIR, exist_ok=True)

class EnhancedFPSCalculator:
    def __init__(self, window_size=60):
        self.timestamps = deque(maxlen=window_size)
        self.smooth_fps = 0.0
    
    def update(self, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()
        self.timestamps.append(timestamp)
        if len(self.timestamps) < 2:
            return 0.0
        time_span = self.timestamps[-1] - self.timestamps[0]
        self.smooth_fps = (len(self.timestamps)-1) / max(0.001, time_span)
        return self.smooth_fps

class VideoReceiverThread(QThread):
    update_frame_signal = pyqtSignal(object)
    update_stats_signal = pyqtSignal(dict)
    update_status_signal = pyqtSignal(str)
    send_control_signal = pyqtSignal(dict)
    update_imu_signal = pyqtSignal(dict)
    connection_lost = pyqtSignal()

    def __init__(self, host, port, app):
        super().__init__()
        self.is_recording = False
        self.host = host
        self.port = port
        self.app = app
        self.server_socket = None
        self.control_socket = None
        self.frame_queue = deque(maxlen=10)
        self.running = Event()
        self.running.set()

        self.current_meta = {}
        
        try:
            self.jpeg = TurboJPEG("C:/libjpeg-turbo-gcc64/bin/libturbojpeg.dll")
            self.turbo_enabled = True
        except:
            self.turbo_enabled = False
            print("⚠️ TurboJPEG不可用，使用OpenCV软解码")

        self.stat_window = deque(maxlen=30)
        self.last_report = time.monotonic()
        self.fps_tool = EnhancedFPSCalculator()

    def run(self):
        self.server_socket = self.create_socket(self.port)
        self.imu_socket = self.create_imu_socket(self.port+2)
        self.control_socket = self.create_control_socket(self.port+1)
        self.transfer_socket = self.create_socket(self.port+3)
        print(f"数据通道: {self.port} | imu通道: {self.port+1} | 控制通道: {self.port+2} | 传输通道: {self.port+3}")
        Thread(target=self.accept_connection, daemon=True).start()
        Thread(target=self.decode_worker, daemon=True).start()
        Thread(target=self.monitor_worker, daemon=True).start()
        Thread(target=self.receive_imu_data, daemon=True).start()
        # Thread(target=self.receive_files, daemon=True).start()


    def receive_files(self, save_dir):
        import zipfile
        import tempfile
        from socket import error as socket_error
        from tqdm import tqdm  # 导入进度条库

        # 确保保存目录存在
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        print(f"[接收端] 准备接收文件到 {save_dir}")
        temp_zip = None  # 临时ZIP文件句柄
        s = self.transfer_conn  # 传输套接字
        
        def read_exact(n):
            """从socket精确读取n个字节"""
            data = bytearray()
            while len(data) < n:
                packet = s.recv(n - len(data))
                if not packet:
                    raise ConnectionAbortedError("连接中断")
                data.extend(packet)
            return bytes(data)

        try:
            # 创建临时ZIP文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_zip:
                zip_filename = temp_zip.name
                print(f"[接收端] 准备写入临时文件: {zip_filename}")

                # 开始接收数据块
                print("[接收端] 等待数据传输...")
                with tqdm(unit='B', unit_scale=True, unit_divisor=1024, desc="接收进度") as progress_bar:  # 创建进度条
                    while True:
                        # 读取块长度头 (4字节大端序)
                        header = read_exact(4)
                        chunk_len = struct.unpack(">I", header)[0]

                        # 终止条件：收到长度为0的块
                        if chunk_len == 0:
                            print("[接收端] 收到传输结束信号")
                            break

                        # 读取实际数据块
                        chunk_data = read_exact(chunk_len)
                        temp_zip.write(chunk_data)
                        progress_bar.update(len(chunk_data))  # 更新进度条

            # 解压ZIP文件
            print(f"[接收端] 开始解压文件到 {save_dir}")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(save_dir)
                print(f"[接收端] 解压完成，共 {len(zip_ref.namelist())} 个文件")

        except (struct.error, socket_error, zipfile.BadZipFile) as e:
            print(f"[接收端] 文件接收失败: {str(e)}")
            raise  
        finally:
            # 确保删除临时文件
            if zip_filename and os.path.exists(zip_filename):
                try:
                    os.remove(zip_filename)
                    print(f"[接收端] 已清理临时文件: {zip_filename}")
                except OSError as e:
                    print(f"[接收端] 清理临时文件失败: {str(e)}")
            # 关闭传输连接
            if s:
                s.close()
                print("[接收端] 传输连接已关闭")

    def _recv_data(self, sock):
        # 接收数据直到完整接收到
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if len(chunk) < 4096:
                break
        return b''.join(chunks)

    def create_imu_socket(self, port=None):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024*1024)  # 增大至1MB
        sock.bind((self.host, port))
        return sock
        
    def create_socket(self, port=None):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4*1024*1024)
        sock.bind((self.host, port))
        sock.listen(1)
        sock.settimeout(2)
        return sock

    def create_control_socket(self, port=None):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, port))
        server_socket.listen(1)
        return server_socket
    
    def accept_connection(self):
        data_conn = None
        control_conn = None
        imu_conn = None
        transfer_conn = None
        while self.running.is_set():
            try:
                if not data_conn:
                    data_conn, addr = self.server_socket.accept()
                    data_conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    self.update_status_signal.emit(f"数据通道连接: {addr}")
                
                if not control_conn:
                    control_conn, addr = self.control_socket.accept()
                    self.update_status_signal.emit(f"控制通道连接: {addr}")
                    self.control_conn = control_conn
                
                if not transfer_conn:
                    transfer_conn, addr = self.transfer_socket.accept()
                    self.update_status_signal.emit(f"传输通道连接: {addr}")
                    self.transfer_conn = transfer_conn
                
                if data_conn and control_conn:
                    self.handle_data_stream(data_conn)

            except socket.timeout:
                continue
            except Exception as e:
                self.update_status_signal.emit(f"连接错误: {str(e)}")
                data_conn = control_conn = None
                time.sleep(1)
                
    def receive_imu_data(self):
        while self.running.is_set():
            try:
                data, addr = self.imu_socket.recvfrom(1024)  # 接收数据
                # print(f"接收到IMU数据: {data}")
                if len(data) == struct.calcsize('!ffffffffff'):
                    # 解析数据，与客户端打包格式一致
                    accel_x, accel_y, accel_z, \
                    gyro_x, gyro_y, gyro_z, \
                    mag_x, mag_y, mag_z, \
                    timestamp = struct.unpack('!ffffffffff', data)
                    # print(struct.unpack('!ffffffffff', data))
                    # 处理IMU数据（示例：打印或存储）
                    # print(timestamp)
                    timestamp = time.time()
                    self.update_imu_signal.emit({
                        'timestamp': timestamp,
                        'accel': [accel_x, accel_y, accel_z],
                        'gyro': [gyro_x, gyro_y, gyro_z],
                        'mag': [mag_x, mag_y, mag_z]
                    })
            except socket.timeout:
                continue
            except Exception as e:
                print(f"接收IMU数据异常: {e}")
                
    def handle_data_stream(self, conn):
        buffer = bytearray()
        header_size = struct.calcsize(">L")
        last_valid = time.monotonic()
        
        try:
            while self.running.is_set():
                conn.settimeout(2)
                chunk = conn.recv(65535)
                if not chunk:
                    raise ConnectionResetError("客户端断开")
                
                buffer.extend(chunk)
                last_valid = time.monotonic()

                while len(buffer) >= header_size:
                    msg_size = struct.unpack(">L", buffer[:header_size])[0]
                    total_size = header_size + msg_size
                    
                    if len(buffer) < total_size:
                        break
                    
                    frame_data = bytes(buffer[header_size:total_size])
                    buffer = buffer[total_size:]
                    self.frame_queue.append((time.monotonic(), frame_data))

                if time.monotonic() - last_valid > 5:
                    raise TimeoutError("数据流超时")

        except (TimeoutError, ConnectionResetError) as e:
            self.update_status_signal.emit(f"连接异常: {str(e)}")
        finally:
            self.cleanup_connection(conn)

    def decode_worker(self):
        while self.running.is_set():
            if not self.frame_queue:
                time.sleep(0.001)
                continue
            
            recv_time, frame_data = self.frame_queue.popleft()
            
            try:
                payload = pickle.loads(frame_data)
                if not isinstance(payload, dict):
                    raise ValueError("无效数据格式")
                
                compressed = payload['frame']
                client_fps = payload.get('fps', 0)
                timestamp = payload.get('timestamp', 0)
                quality = payload.get('quality', 75)

                decode_start = time.perf_counter()
                if self.turbo_enabled:
                    try:
                        frame = self.jpeg.decode(compressed, flags=TJFLAG_FASTDCT)
                    except:
                        frame = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_COLOR)
                else:
                    frame = cv2.imdecode(np.frombuffer(compressed, np.uint8), cv2.IMREAD_COLOR)
                # print(frame.shape)
                decode_time = time.perf_counter() - decode_start
                e2e_delay = time.time() - timestamp if timestamp > 0 else 0

                self.stat_window.append({
                    'timestamp': time.perf_counter(),
                    'decode_time': decode_time,
                    'e2e_delay': e2e_delay,
                    'size': len(compressed)
                })

                if time.perf_counter() - self.last_report > 0.033:
                    self.update_frame_signal.emit(frame)
                    self.last_report = time.perf_counter()

            except (pickle.UnpicklingError, KeyError) as e:
                print(f"数据解析失败: {str(e)}")
                self.send_control({'action': 'reset'})
            except Exception as e:
                print(f"解码异常: {str(e)}")
                self.send_control({'action': 'quality', 'value': -10})

    def monitor_worker(self):
        while self.running.is_set():
            if len(self.stat_window) < 2:
                time.sleep(1)
                continue
            
            time_span = self.stat_window[-1]['timestamp'] - self.stat_window[0]['timestamp']
            total_frames = len(self.stat_window)
            
            stats = {
                'fps': total_frames / max(0.001, time_span),
                'avg_decode': sum(d['decode_time'] for d in self.stat_window) / total_frames,
                'avg_delay': sum(d['e2e_delay'] for d in self.stat_window) / total_frames,
                'bitrate': sum(d['size'] for d in self.stat_window)*8 / time_span / 1e6
            }
            
            self.update_stats_signal.emit(stats)
            time.sleep(1)

    def send_control(self, command):
        try:
            data = json.dumps(command).encode()
            header = struct.pack(">I", len(data))
            self.control_conn.sendall(header + data)
        except Exception as e:
            print(f"控制指令发送失败: {str(e)}")

    def stop(self):
        print("正在停止服务器...")
        self.running.clear()
        self.cleanup_connection(self.server_socket, self.control_socket)
        self.update_status_signal.emit("服务器已安全停止")

    def cleanup_connection(self, *conns):
        print("正在关闭连接...")
        for conn in conns:
            if conn:
                try:
                    conn.shutdown(socket.SHUT_RDWR)
                except Exception as e:
                    print(f"关闭连接时出错: {str(e)}")
                finally:
                    conn.close()
                    
class VideoApp(QWidget):
    def __init__(self, gesture_config_path='./data/gesture_config.json'):
        super().__init__()
        self.setWindowTitle("视频采集服务端")
        self.setGeometry(100, 100, 1400, 1000)  # 增大窗口大小
        self.is_recording = False
        self.init_ui()
        self.init_video()
        self.init_participant_info()
        # self.video_thread.update_imu_signal.connect(self.update_imu_charts)
        self.data_buffer = {
            'accel': deque(maxlen=200),
            'gyro': deque(maxlen=200),
            'mag': deque(maxlen=200)
        }
        self.video_count = 0
        self.fps_tool = EnhancedFPSCalculator()
        self.plot_counter = 0  # 初始化计数器

        # 添加定时器，每隔100ms更新一次图表
        self.chart_update_timer = QTimer(self)
        self.chart_update_timer.timeout.connect(self.async_update_plots)
        self.chart_update_timer.start(50)

        self.init_gestures(gesture_config_path)
        self.communicate = Communicate()
        self.communicate.update_image_signal.connect(self.update_image)

        # 添加手势更新计时器
        self.remaining_time = self.update_interval
        self.gesture_update_timer = QTimer(self)
        self.gesture_update_timer.timeout.connect(self.update_gesture_and_countdown)
        self.gesture_update_timer.start(100)
        
        # 启动手势更新进程
        self.gesture_process = threading.Thread(target=self.gesture_image_process)
        self.gesture_process.start()
    
    def init_participant_info(self):
        self.participant_info = {
            'name': '',
            'purpose': '',
            'start_time': '',
            'file_path': '',
            'resolution': '',
            'codec': '',
            'start_timestamp': ''
        }
    
    def init_gestures(self, gesture_config_path):
        self.gesture_images = []
        with open(gesture_config_path, 'r') as f:
            gesture_config = json.load(f)
        for gesture in gesture_config['gestures']:
            gesture_name = gesture['name']
            gesture_path = gesture['image']
            self.gesture_images.append((gesture_name, gesture_path))
        self.gesture_images.sort(key=lambda x: x[0])
        self.session_count = gesture_config['session_count']
        self.repeat_count_per_gesture = gesture_config['repeat_count_per_gesture']
        self.random_gesture_count = gesture_config['random_gesture_count']
        self.gesture_count = len(self.gesture_images)
        self.gesture_display_indexs = [self.get_gesture_display_indexs() for _ in range(self.session_count)]
        self.gesture_display_index = -1
        self.current_gesture_index = -1
        self.current_session_index = 0
        self.update_interval = gesture_config['update_interval']
        self.gesture_queue = Queue()

    def update_gesture_and_countdown(self):
        if self.is_recording:
            self.remaining_time -= 0.1
            self.current_gesture_index = self.gesture_display_indexs[self.current_session_index][self.gesture_display_index]
            current_text = f"当前手势: {self.gesture_images[self.current_gesture_index][0]}"
            text = f"，倒计时: {self.remaining_time:.1f}秒"
            current_text = current_text + text
            text = ""
            if not self.gesture_display_index == len(self.gesture_display_indexs[self.current_session_index]) - 1:
                next_gesture_index = self.gesture_display_indexs[self.current_session_index][self.gesture_display_index+1]
                next_gesture_name, next_gesture_path = self.gesture_images[next_gesture_index]
                text += f"下一个手势: {next_gesture_name}"
                text += f", 还有{len(self.gesture_display_indexs[self.current_session_index]) - self.gesture_display_index - 1}个手势"
                pixmap = QPixmap(next_gesture_path)
                self.next_image_label.setPixmap(pixmap.scaled(self.next_image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                text += f", 结束后本session结束"
                self.next_image_label.clear()  # 清除下一个手势图片
            self.next_gesture_label.setText(text)  # 将倒计时信息放在第二个手势标签处
            self.current_gesture_label.setText(current_text)  # 将当前手势信息放在第一个手势标签处
            self.current_gesture_label.setStyleSheet("QLabel { font-size: 24pt; color: red; }")
            self.next_gesture_label.setStyleSheet("QLabel { font-size: 24pt; color: red; }")
            if self.remaining_time <= 0:
                self.change_gesture_image()
                self.remaining_time = self.update_interval
        else:
            self.next_gesture_label.setText("")  # 清除倒计时信息
            self.next_image_label.clear()  # 清除下一个手势图片

    def get_gesture_display_indexs(self):
        import random
        gesture_display_indexs = []
        indexs = []
        # 第一部分
        indexs = [i for i in range(self.gesture_count) for _ in range(self.repeat_count_per_gesture)]
        random.shuffle(indexs)
        gesture_display_indexs.extend(indexs)
        # 第二部分
        indexs = [random.randint(0, self.gesture_count-1) for _ in range(self.random_gesture_count)]
        random.shuffle(indexs)
        gesture_display_indexs.extend(indexs)
        return gesture_display_indexs

    def init_ui(self):
        self.layout = QVBoxLayout(self)  # 直接设置为主布局

        # ===== 视频区域 (70%) =====
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.video_label, stretch=7)  # 比例参数

        # ===== 综合图表区域 (30%) =====
        charts_widget = QWidget()
        charts_layout = QHBoxLayout(charts_widget)
        
        # 加速度图表
        self.accel_fig = Figure()
        self.accel_ax = self.accel_fig.add_subplot(111)
        self.accel_canvas = FigureCanvas(self.accel_fig)
        self.accel_ax.set_title("Acceleration")
        self.accel_ax.set_xlabel("Time")
        self.accel_ax.set_ylabel("m/s²")
        self.accel_line_x, = self.accel_ax.plot([], [], 'r-', label='X')
        self.accel_line_y, = self.accel_ax.plot([], [], 'g-', label='Y')
        self.accel_line_z, = self.accel_ax.plot([], [], 'b-', label='Z')
        self.accel_ax.legend()
        self.accel_ax.set_xlim(0, 100)
        
        # 角速度图表
        self.gyro_fig = Figure()
        self.gyro_ax = self.gyro_fig.add_subplot(111)
        self.gyro_canvas = FigureCanvas(self.gyro_fig)
        self.gyro_ax.set_title("Gyroscope")
        self.gyro_ax.set_xlabel("Time")
        self.gyro_ax.set_ylabel("deg/s")
        self.gyro_line_x, = self.gyro_ax.plot([], [], 'r-', label='X')
        self.gyro_line_y, = self.gyro_ax.plot([], [], 'g-', label='Y')
        self.gyro_line_z, = self.gyro_ax.plot([], [], 'b-', label='Z')
        self.gyro_ax.legend()
        self.gyro_ax.set_xlim(0, 100)
        
        charts_layout.addWidget(self.accel_canvas)
        charts_layout.addWidget(self.gyro_canvas)
        self.layout.addWidget(charts_widget, stretch=5)
        
        # ===== 手势提示图片展示 =====
        gesture_images_widget = QWidget()
        gesture_images_layout = QHBoxLayout(gesture_images_widget)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { font-size: 24pt; color: red; }")
        gesture_images_layout.addWidget(self.image_label, stretch=1)  # 当前手势图片

        # 新增下一个手势图片展示
        self.next_image_label = QLabel()
        self.next_image_label.setAlignment(Qt.AlignCenter) 
        self.next_image_label.setStyleSheet("QLabel { font-size: 24pt; color: red; }")
        gesture_images_layout.addWidget(self.next_image_label, stretch=1)  # 下一个手势图片

        self.layout.addWidget(gesture_images_widget, stretch=7)  # 将手势图片区域添加到主布局

        # 添加手势标签
        gesture_labels_widget = QWidget()
        gesture_labels_layout = QHBoxLayout(gesture_labels_widget)

        self.current_gesture_label = QLabel("")
        self.current_gesture_label.setAlignment(Qt.AlignCenter)
        gesture_labels_layout.addWidget(self.current_gesture_label, stretch=1)

        self.next_gesture_label = QLabel("下一个手势")
        self.next_gesture_label.setAlignment(Qt.AlignCenter)
        gesture_labels_layout.addWidget(self.next_gesture_label, stretch=1)

        self.layout.addWidget(gesture_labels_widget, stretch=1)

        # ===== 添加填写受试者信息的按钮 =====
        self.info_button = QPushButton("填写受试者信息")
        self.info_button.clicked.connect(self.fill_participant_info)  # 连接到处理方法
        self.layout.addWidget(self.info_button)  # 将按钮添加到布局中

        # 初始化数据存储
        self.accel_data = {'x': [], 'y': [], 'z': []}
        self.gyro_data = {'x': [], 'y': [], 'z': []}
        # 初始化数据存储（使用双缓冲）
        self.accel_buffer = {'x': [], 'y': [], 'z': []}
        self.gyro_buffer = {'x': [], 'y': [], 'z': []}
        self.data_lock = threading.Lock()  # 数据锁
        
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        
        # 统计信息
        self.fps_label = QLabel("FPS: - | 延迟: -")
        control_layout.addWidget(self.fps_label)
        
        # 控制按钮
        self.start_btn = QPushButton("开始录制")
        self.stop_btn = QPushButton("停止录制") 
        self.del_btn = QPushButton("删除视频")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.del_btn)
        self.start_btn.clicked.connect(self.start_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        self.del_btn.clicked.connect(self.delete_video)
        
        
        # 状态栏
        self.status_bar = QStatusBar()
        control_layout.addWidget(self.status_bar)
        
        self.layout.addWidget(control_widget)  # 第五行
    
    def init_video(self):
        self.video_thread = VideoReceiverThread('0.0.0.0', 9999, self)
        self.video_thread.update_frame_signal.connect(self.update_frame)
        self.video_thread.update_stats_signal.connect(self.update_stats)
        self.video_thread.update_status_signal.connect(self.update_status)
        self.video_thread.connection_lost.connect(self.on_connection_lost)
        self.video_thread.update_imu_signal.connect(self.update_imu_charts)
        self.video_thread.start()
    
    def update_imu_charts(self, data):
        # 仅做数据缓存，不进行绘图操作
        with self.data_lock:
            self.accel_buffer['x'].append(data['accel'][0])
            self.accel_buffer['y'].append(data['accel'][1])
            self.accel_buffer['z'].append(data['accel'][2])
            
            self.gyro_buffer['x'].append(data['gyro'][0])
            self.gyro_buffer['y'].append(data['gyro'][1])
            self.gyro_buffer['z'].append(data['gyro'][2])
    
    def async_update_plots(self):
        # 批量处理数据更新
        max_points = 100
        accel_data = {'x': [], 'y': [], 'z': []}
        gyro_data = {'x': [], 'y': [], 'z': []}
        
        # 交换数据缓冲区
        with self.data_lock:
            # 移动数据到处理队列
            accel_data = self.accel_buffer
            gyro_data = self.gyro_buffer
            self.accel_buffer = {'x': [], 'y': [], 'z': []}
            self.gyro_buffer = {'x': [], 'y': [], 'z': []}
        
        # 更新加速度数据（保留最近max_points个点）
        self.accel_data['x'] = (self.accel_data['x'] + accel_data['x'])[-max_points:]
        self.accel_data['y'] = (self.accel_data['y'] + accel_data['y'])[-max_points:]
        self.accel_data['z'] = (self.accel_data['z'] + accel_data['z'])[-max_points:]
        
        # 更新角速度数据（保留最近max_points个点）
        self.gyro_data['x'] = (self.gyro_data['x'] + gyro_data['x'])[-max_points:]
        self.gyro_data['y'] = (self.gyro_data['y'] + gyro_data['y'])[-max_points:]
        self.gyro_data['z'] = (self.gyro_data['z'] + gyro_data['z'])[-max_points:]
        
        # 生成时间序列
        time_points = list(range(len(self.accel_data['x'])))
        
        # 更新加速度图表（使用快速更新方法）
        self.accel_line_x.set_data(time_points, self.accel_data['x'])
        self.accel_line_y.set_data(time_points, self.accel_data['y'])
        self.accel_line_z.set_data(time_points, self.accel_data['z'])
        
        # 更新角速度图表（使用快速更新方法）
        self.gyro_line_x.set_data(time_points, self.gyro_data['x'])
        self.gyro_line_y.set_data(time_points, self.gyro_data['y'])
        self.gyro_line_z.set_data(time_points, self.gyro_data['z'])
        
        # 仅更新必要区域
        self.accel_ax.draw_artist(self.accel_ax.patch)
        self.accel_ax.draw_artist(self.accel_line_x)
        self.accel_ax.draw_artist(self.accel_line_y)
        self.accel_ax.draw_artist(self.accel_line_z)
        
        self.gyro_ax.draw_artist(self.gyro_ax.patch)
        self.gyro_ax.draw_artist(self.gyro_line_x)
        self.gyro_ax.draw_artist(self.gyro_line_y)
        self.gyro_ax.draw_artist(self.gyro_line_z)
        
        # 更新画布
        self.accel_canvas.update()
        self.gyro_canvas.update()
        
        # 自动调整范围（每10次刷新调整一次）
        if self.plot_counter % 10 == 0:
            self.accel_ax.relim()
            self.accel_ax.autoscale_view(scaley=True)
            self.gyro_ax.relim()
            self.gyro_ax.autoscale_view(scaley=True)
            
            x_min = max(0, len(time_points)-100)
            self.accel_ax.set_xlim(x_min, len(time_points))
            self.gyro_ax.set_xlim(x_min, len(time_points))
            
        self.plot_counter += 1
    
    def update_stats(self, stats):
        text = (f"FPS: {stats['fps']:.1f} | "
                f"解码: {stats['avg_decode']*1000:.1f}ms | "
                f"延迟: {stats['avg_delay']:.2f}s | "
                f"码率: {stats['bitrate']:.2f}Mbps")
        self.fps_label.setText(text)

    def update_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_img).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def update_status(self, msg):
        self.status_bar.showMessage(msg)

    def gesture_image_process(self):
        while True:
            # 从队列中获取手势索引
            current_gesture_index, current_session_index = self.gesture_queue.get()
            if current_gesture_index is None:  # 结束信号
                break

            # 处理手势图像更新
            gesture_name, gesture_path = self.gesture_images[current_gesture_index]
            self.communicate.update_image_signal.emit(gesture_name, gesture_path)

    def change_gesture_image(self):
        if self.gesture_images:
            self.gesture_display_index += 1
            if self.gesture_display_index >= len(self.gesture_display_indexs[self.current_session_index]): # 当前session的手势采集完成
                self.video_thread.send_control({'action': 'pause_recording'})
                self.ask_if_continue()
                self.video_thread.send_control({'action': 'resume_recording'})
                self.current_session_index += 1
                self.gesture_display_index = 0
                if self.current_session_index == self.session_count: # 所有session采集完成
                    print("所有session采集完成")
                    # confirm to save or not
                    reply = QMessageBox.question(self, "提示", "是否保存当前受试者数据？", QMessageBox.Yes | QMessageBox.No)
                    self.stop_recording(reply)
                    if reply == QMessageBox.Yes:
                        self.save_participant_data()
                    return  # 结束条件
            # 将当前手势索引发送到进程
            self.current_gesture_index = self.gesture_display_indexs[self.current_session_index][self.gesture_display_index]
            self.gesture_queue.put((self.current_gesture_index, self.current_session_index))

    def ask_if_continue(self):
        # 创建一个消息框
        self.continue_dialog = QMessageBox(self)
        self.continue_dialog.setWindowTitle("继续")
        self.continue_dialog.setText(f"当前session已完成，请摘下手环并重新带上，休息指定时间后进行下一个session。还有{self.session_count - self.current_session_index - 1}个session")
        self.continue_dialog.setStandardButtons(QMessageBox.Yes)
        
        # 显示消息框并等待用户选择
        reply = self.continue_dialog.exec_()

    def save_participant_data(self):
        
        # 保存元数据
        participant_dir = os.path.join(DATA_DIR, self.participant_info['name'])
        os.makedirs(participant_dir, exist_ok=True)
        participant_info_file = os.path.join(participant_dir, 'participant_info.json')
        # with open(participant_info_file, 'w', encoding='utf-8') as f:
        #     json.dump(self.participant_info, f, indent=2, ensure_ascii=False)
        
        # 保存手势数据
        for session_index, session_data in enumerate(self.gesture_display_indexs):
            session_gestures = []
            for gesture_index in session_data:
                gesture_data = {
                    'gesture_name': self.gesture_images[gesture_index][0],
                    'duration': self.update_interval
                }
                session_gestures.append(gesture_data)
            session_file = os.path.join(participant_dir, 'gestures', f'session_{session_index}.json')
            os.makedirs(os.path.dirname(session_file), exist_ok=True)
            with open(session_file, 'w', encoding='utf-8') as f:
                json.dump(session_gestures, f, indent=2, ensure_ascii=False)
        
        # 保存video数据
        print(f"受试者数据已保存: {participant_dir}")
        QMessageBox.information(self, "成功", "受试者数据已保存")
        
    def update_image(self, gesture_name, gesture_path):
        pixmap = QPixmap(gesture_path)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        # self.gesture_info_label.setText(f"当前手势: {gesture_name}")

    def on_connection_lost(self):
        QMessageBox.warning(self, "连接中断", "客户端连接已断开，等待重新连接...")

    def start_recording(self):
        if self.is_recording:
            return
        if self.participant_info['name'] == '':
            QMessageBox.warning(self, "警告", "必须先填写被试者信息")
            return
        print("开始录制视频...")
        # 采集元数据
        os.makedirs(DATA_DIR, exist_ok=True)

        # 生成元数据
        basename = datetime.now().strftime(f"{self.participant_info['name']}_%Y%m%d_%H%M%S")
        video_filename = f"{basename}_video.avi"
        imu_filename = f"{basename}_imu.csv"
        
        video_path = os.path.join(DATA_DIR, self.participant_info['name'], video_filename)
        imu_path = os.path.join(DATA_DIR, self.participant_info['name'], imu_filename)
        self.participant_info['video_path'] = video_path
        self.participant_info['imu_path'] = imu_path
        
        command = {
            'action': 'start_recording',
            'participant_info': self.participant_info
        }
        self.video_thread.send_control(command)
        self.is_recording = True
        self.change_gesture_image()

    def stop_recording(self, reply=None):
        '''停止录制，但是不发送'''
        if not self.is_recording:
            return
        self.is_recording = False
        if reply == QMessageBox.Yes:
            command = {
                'action': 'stop_recording',
                'reply': True
            }
            self.video_thread.send_control(command)
            threading.Thread(target=self.video_thread.receive_files, args=(os.path.join(DATA_DIR, self.participant_info['name']),), daemon=True).start()
        else:
            command = {
                'action': 'stop_recording',
                'reply': False
            }
            self.video_thread.send_control(command)
            self.update_status("视频已保存")

    def transfer_data(self):
        command = {
            'action': 'transfer_data'
        }
        self.video_thread.send_control(command)

    def delete_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "选择要删除的视频", DATA_DIR, "视频文件 (*.avi)")
        if file_name:
            try:
                os.remove(file_name)
                meta_file = os.path.splitext(file_name)[0] + ".meta.json"
                if os.path.exists(meta_file):
                    os.remove(meta_file)
                self.video_thread.send_control({
                    'action': 'delete_video',
                    'filename': os.path.basename(file_name)
                })
                QMessageBox.information(self, "成功", "视频删除成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"删除失败: {str(e)}")

    def closeEvent(self, event):
        # 结束进程
        self.gesture_queue.put((None, None))
        self.gesture_process.join()  # 等待进程结束
        if self.is_recording:
            self.stop_recording()
        self.video_thread.stop()
        event.accept()

    def fill_participant_info(self):
        name, ok1 = QInputDialog.getText(self, '信息录入', '被采集者姓名:', QLineEdit.Normal)
        purpose, ok2 = QInputDialog.getText(self, '信息录入', '采集目的:', QLineEdit.Normal)

        if ok1 and ok2 and name:
            self.participant_info['name'] = name
            self.participant_info['purpose'] = purpose
            QMessageBox.information(self, "成功", "受试者信息已保存")
        else:
            QMessageBox.warning(self, "警告", "必须填写被采集者姓名")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VideoApp()
    window.show()
    sys.exit(app.exec_())