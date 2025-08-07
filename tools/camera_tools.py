import cv2
import os
import time
import threading
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class CameraManager:
    """摄像头管理器"""
    
    def __init__(self):
        self.camera = None
        self.is_recording = False
        self.recording_thread = None
        self.output_path = None
        self.camera_index = 0
        
    def get_available_cameras(self) -> list:
        """获取可用的摄像头列表"""
        available_cameras = []
        for i in range(10):  # 检查前10个摄像头索引
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()
        return available_cameras
    
    def open_camera(self, camera_index: int = 0) -> bool:
        """打开摄像头"""
        try:
            if self.camera is not None:
                self.close_camera()
            
            self.camera = cv2.VideoCapture(camera_index)
            self.camera_index = camera_index
            
            if not self.camera.isOpened():
                logger.error(f"无法打开摄像头 {camera_index}")
                return False
                
            logger.info(f"成功打开摄像头 {camera_index}")
            return True
            
        except Exception as e:
            logger.error(f"打开摄像头失败: {e}")
            return False
    
    def close_camera(self):
        """关闭摄像头"""
        try:
            if self.camera is not None:
                # 如果正在录制，先停止录制
                if self.is_recording:
                    self.stop_recording()
                
                self.camera.release()
                self.camera = None
                logger.info("摄像头已关闭")
                return {"success": True, "message": "摄像头已关闭"}
            else:
                return {"success": False, "error": "摄像头未打开"}
        except Exception as e:
            logger.error(f"关闭摄像头失败: {e}")
            return {"success": False, "error": str(e)}
    
    def auto_close_camera(self, delay_seconds: int = 5):
        """自动关闭摄像头（延迟关闭）"""
        try:
            import threading
            import time
            
            def delayed_close():
                time.sleep(delay_seconds)
                self.close_camera()
            
            # 在新线程中延迟关闭
            close_thread = threading.Thread(target=delayed_close, daemon=True)
            close_thread.start()
            logger.info(f"摄像头将在 {delay_seconds} 秒后自动关闭")
            
        except Exception as e:
            logger.error(f"设置自动关闭摄像头失败: {e}")
    
    def take_photo(self, save_path: Optional[str] = None, auto_close: bool = True) -> Dict[str, Any]:
        """拍照"""
        try:
            if self.camera is None:
                if not self.open_camera():
                    return {"success": False, "error": "无法打开摄像头"}
            
            ret, frame = self.camera.read()
            if not ret:
                return {"success": False, "error": "无法获取图像"}
            
            if save_path is None:
                # 创建临时文件
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(tempfile.gettempdir(), f"photo_{timestamp}.jpg")
            
            # 确保目录存在
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 保存图片
            cv2.imwrite(save_path, frame)
            
            # 获取图片信息
            height, width = frame.shape[:2]
            
            result = {
                "success": True,
                "file_path": save_path,
                "width": width,
                "height": height,
                "timestamp": datetime.now().isoformat()
            }
            
            # 如果启用自动关闭，延迟关闭摄像头
            if auto_close:
                self.auto_close_camera(delay_seconds=3)
                result["message"] = "拍照完成，摄像头将在3秒后自动关闭"
            
            return result
            
        except Exception as e:
            logger.error(f"拍照失败: {e}")
            return {"success": False, "error": str(e)}
    
    def start_recording(self, output_path: str, duration: int = 10) -> Dict[str, Any]:
        """开始录制视频"""
        try:
            if self.camera is None:
                if not self.open_camera():
                    return {"success": False, "error": "无法打开摄像头"}
            
            if self.is_recording:
                return {"success": False, "error": "正在录制中"}
            
            # 获取视频参数
            fps = int(self.camera.get(cv2.CAP_PROP_FPS))
            if fps == 0:
                fps = 30  # 默认帧率
            
            width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # 确保目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:  # 只有当目录不为空时才创建
                os.makedirs(output_dir, exist_ok=True)
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            self.is_recording = True
            self.output_path = output_path
            
            # 在新线程中录制
            def record_video():
                start_time = time.time()
                try:
                    while self.is_recording and (time.time() - start_time) < duration:
                        ret, frame = self.camera.read()
                        if ret:
                            out.write(frame)
                        else:
                            break
                        time.sleep(1/fps)
                finally:
                    out.release()
                    self.is_recording = False
                    logger.info(f"视频录制完成: {output_path}")
            
            self.recording_thread = threading.Thread(target=record_video)
            self.recording_thread.start()
            
            return {
                "success": True,
                "output_path": output_path,
                "duration": duration,
                "fps": fps,
                "resolution": f"{width}x{height}"
            }
            
        except Exception as e:
            logger.error(f"开始录制失败: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_recording(self) -> Dict[str, Any]:
        """停止录制视频"""
        try:
            if not self.is_recording:
                return {"success": False, "error": "当前没有在录制"}
            
            self.is_recording = False
            
            if self.recording_thread:
                self.recording_thread.join(timeout=5)
            
            return {
                "success": True,
                "output_path": self.output_path,
                "message": "录制已停止"
            }
            
        except Exception as e:
            logger.error(f"停止录制失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_camera_info(self) -> Dict[str, Any]:
        """获取摄像头信息"""
        try:
            if self.camera is None:
                return {"success": False, "error": "摄像头未打开"}
            
            info = {
                "camera_index": self.camera_index,
                "is_opened": self.camera.isOpened(),
                "is_recording": self.is_recording,
                "width": int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": int(self.camera.get(cv2.CAP_PROP_FPS)),
                "brightness": self.camera.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.camera.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.camera.get(cv2.CAP_PROP_SATURATION)
            }
            
            return {"success": True, "info": info}
            
        except Exception as e:
            logger.error(f"获取摄像头信息失败: {e}")
            return {"success": False, "error": str(e)}

# 全局摄像头管理器实例
camera_manager = CameraManager()
