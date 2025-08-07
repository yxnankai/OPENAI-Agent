import os
import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
import tempfile
from typing import Dict, List, Any, Optional
import logging
from transformers import pipeline, AutoImageProcessor, AutoModelForImageClassification
from ultralytics import YOLO
import requests
from .model_manager import model_config_manager

logger = logging.getLogger(__name__)

class LocalModelManager:
    """本地模型管理器"""
    
    def __init__(self):
        self.models = {}
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型配置管理器
        self.config_manager = model_config_manager
        
        # 初始化常用模型
        self._init_models()
    
    def _init_models(self):
        """初始化常用模型"""
        try:
            # 图像分类模型
            self.models['image_classification'] = pipeline(
                "image-classification",
                model="microsoft/resnet-50",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("图像分类模型加载成功")
        except Exception as e:
            logger.warning(f"图像分类模型加载失败: {e}")
        
        try:
            # 从本地配置加载YOLOv8模型
            self._load_yolo_models()
        except Exception as e:
            logger.warning(f"YOLO模型加载失败: {e}")
        
        try:
            # 人脸检测模型
            self.models['face_detection'] = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            logger.info("人脸检测模型加载成功")
        except Exception as e:
            logger.warning(f"人脸检测模型加载失败: {e}")
    
    def _load_yolo_models(self):
        """从本地配置加载YOLOv8模型"""
        available_models = self.config_manager.get_available_models()
        
        for model_id, model_info in available_models.items():
            if model_info["status"] == "available":
                config = model_info["config"]
                if config["type"] == "yolov8":
                    try:
                        model_path = model_info["file_path"]
                        self.models[f'yolo_{model_id}'] = YOLO(model_path)
                        logger.info(f"YOLOv8模型加载成功: {model_id}")
                    except Exception as e:
                        logger.warning(f"YOLOv8模型加载失败 {model_id}: {e}")
        
        # 如果没有加载到任何本地模型，使用默认模型
        if not any(key.startswith('yolo_') for key in self.models.keys()):
            try:
                self.models['object_detection'] = YOLO('yolov8n.pt')
                logger.info("使用默认YOLOv8n模型")
            except Exception as e:
                logger.warning(f"默认YOLOv8模型加载失败: {e}")
    
    def _get_best_yolo_model(self) -> Optional[YOLO]:
        """获取最佳的YOLO模型"""
        # 按优先级选择模型：yolov8n > yolov8s > yolov8m > yolov8l > yolov8x
        model_priority = ['yolo_yolov8n', 'yolo_yolov8s', 'yolo_yolov8m', 'yolo_yolov8l', 'yolo_yolov8x']
        
        for model_key in model_priority:
            if model_key in self.models:
                return self.models[model_key]
        
        # 如果没有找到本地模型，使用默认的object_detection
        return self.models.get('object_detection')
    
    def classify_image(self, image_path: str) -> Dict[str, Any]:
        """图像分类"""
        try:
            if 'image_classification' not in self.models:
                return {"success": False, "error": "图像分类模型未加载"}
            
            # 加载图像
            image = Image.open(image_path)
            
            # 进行分类
            results = self.models['image_classification'](image)
            
            # 格式化结果
            classifications = []
            for result in results:
                classifications.append({
                    "label": result['label'],
                    "confidence": round(result['score'] * 100, 2)
                })
            
            return {
                "success": True,
                "classifications": classifications,
                "top_result": classifications[0] if classifications else None
            }
            
        except Exception as e:
            logger.error(f"图像分类失败: {e}")
            return {"success": False, "error": str(e)}
    
    def detect_objects(self, image_path: str, confidence: float = 0.5, model_id: str = None, 
                      draw_boxes: bool = False, show_confidence: bool = True, 
                      save_annotated: bool = False, mask_threshold: float = 0.5) -> Dict[str, Any]:
        """目标检测"""
        try:
            # 获取YOLO模型
            yolo_model = None
            
            if model_id:
                # 使用指定的模型
                model_key = f'yolo_{model_id}'
                if model_key in self.models:
                    yolo_model = self.models[model_key]
                else:
                    return {"success": False, "error": f"指定的模型 {model_id} 未加载"}
            else:
                # 使用最佳可用模型
                yolo_model = self._get_best_yolo_model()
            
            if yolo_model is None:
                return {"success": False, "error": "目标检测模型未加载"}
            
            # 使用YOLO进行检测
            results = yolo_model(image_path, conf=confidence)
            
            detections = []
            model_name = "未知模型"
            annotated_image_path = None
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 获取边界框坐标
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # 获取类别和置信度
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        
                        # 获取类别名称
                        class_name = result.names[cls]
                        
                        detection_info = {
                            "class": class_name,
                            "confidence": round(conf * 100, 2),
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        }
                        
                        # 如果启用了mask检测
                        if hasattr(box, 'masks') and box.masks is not None:
                            mask = box.masks.data[0].cpu().numpy()
                            detection_info["mask"] = mask.tolist()
                            detection_info["mask_area"] = int(mask.sum())
                        
                        detections.append(detection_info)
            
            # 如果需要绘制标注
            if draw_boxes or save_annotated:
                annotated_image_path = self._draw_detections(image_path, detections, show_confidence)
            
            # 获取模型名称
            if model_id:
                config = self.config_manager.get_model_config(model_id)
                if config:
                    model_name = config["name"]
            
            result = {
                "success": True,
                "detections": detections,
                "total_objects": len(detections),
                "model_used": model_name,
                "confidence_threshold": confidence,
                "parameters": {
                    "draw_boxes": draw_boxes,
                    "show_confidence": show_confidence,
                    "mask_threshold": mask_threshold
                }
            }
            
            if annotated_image_path:
                result["annotated_image"] = annotated_image_path
            
            return result
            
        except Exception as e:
            logger.error(f"目标检测失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _draw_detections(self, image_path: str, detections: List[Dict], show_confidence: bool = True) -> str:
        """绘制检测结果"""
        try:
            import cv2
            import numpy as np
            from datetime import datetime
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # 定义颜色
            colors = [
                (255, 0, 0),    # 蓝色
                (0, 255, 0),    # 绿色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
                (255, 0, 255),  # 洋红
                (0, 255, 255),  # 黄色
                (128, 0, 0),    # 深蓝
                (0, 128, 0),    # 深绿
                (0, 0, 128),    # 深红
                (128, 128, 0)   # 橄榄色
            ]
            
            for i, detection in enumerate(detections):
                bbox = detection["bbox"]
                x1, y1, x2, y2 = bbox
                
                # 选择颜色
                color = colors[i % len(colors)]
                
                # 绘制边界框
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # 准备标签文本
                label = detection["class"]
                if show_confidence:
                    label += f" {detection['confidence']:.1f}%"
                
                # 计算文本大小
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 绘制标签背景
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # 绘制标签文本
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # 如果检测到mask，绘制mask
                if "mask" in detection:
                    mask = np.array(detection["mask"], dtype=np.uint8)
                    # 调整mask大小以匹配图像
                    mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]))
                    # 创建彩色mask
                    colored_mask = np.zeros_like(image)
                    colored_mask[mask_resized > 0] = color
                    # 将mask叠加到图像上
                    image = cv2.addWeighted(image, 1, colored_mask, 0.3, 0)
            
            # 保存标注后的图像
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            annotated_path = f"detection_result_{timestamp}.jpg"
            cv2.imwrite(annotated_path, image)
            
            return annotated_path
            
        except Exception as e:
            logger.error(f"绘制检测结果失败: {e}")
            return None
    
    def detect_faces(self, image_path: str) -> Dict[str, Any]:
        """人脸检测"""
        try:
            if 'face_detection' not in self.models:
                return {"success": False, "error": "人脸检测模型未加载"}
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "error": "无法读取图像"}
            
            # 转换为灰度图
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 检测人脸
            faces = self.models['face_detection'].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            face_detections = []
            for (x, y, w, h) in faces:
                face_detections.append({
                    "bbox": [int(x), int(y), int(w), int(h)],
                    "center": [int(x + w/2), int(y + h/2)]
                })
            
            return {
                "success": True,
                "faces": face_detections,
                "face_count": len(face_detections)
            }
            
        except Exception as e:
            logger.error(f"人脸检测失败: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_image(self, image_path: str, model_id: str = None) -> Dict[str, Any]:
        """综合分析图像（分类+检测+人脸）"""
        try:
            results = {
                "image_path": image_path,
                "analysis": {}
            }
            
            # 图像分类
            classification_result = self.classify_image(image_path)
            if classification_result["success"]:
                results["analysis"]["classification"] = classification_result
            
            # 目标检测
            detection_result = self.detect_objects(image_path, model_id=model_id)
            if detection_result["success"]:
                results["analysis"]["object_detection"] = detection_result
            
            # 人脸检测
            face_result = self.detect_faces(image_path)
            if face_result["success"]:
                results["analysis"]["face_detection"] = face_result
            
            # 生成总结
            summary = self._generate_analysis_summary(results["analysis"])
            results["summary"] = summary
            
            return {
                "success": True,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"图像分析失败: {e}")
            return {"success": False, "error": str(e)}
    
    def _generate_analysis_summary(self, analysis: Dict) -> str:
        """生成分析总结"""
        summary_parts = []
        
        # 分类结果
        if "classification" in analysis:
            top_result = analysis["classification"].get("top_result")
            if top_result:
                summary_parts.append(f"图像主要包含: {top_result['label']} (置信度: {top_result['confidence']}%)")
        
        # 目标检测结果
        if "object_detection" in analysis:
            detections = analysis["object_detection"].get("detections", [])
            if detections:
                object_counts = {}
                for det in detections:
                    obj_class = det["class"]
                    object_counts[obj_class] = object_counts.get(obj_class, 0) + 1
                
                object_summary = ", ".join([f"{count}个{obj}" for obj, count in object_counts.items()])
                summary_parts.append(f"检测到物体: {object_summary}")
        
        # 人脸检测结果
        if "face_detection" in analysis:
            face_count = analysis["face_detection"].get("face_count", 0)
            if face_count > 0:
                summary_parts.append(f"检测到 {face_count} 张人脸")
        
        if not summary_parts:
            return "图像分析完成，但未检测到明显特征"
        
        return " | ".join(summary_parts)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        # 获取本地模型配置信息
        models_summary = self.config_manager.get_models_summary()
        available_models = self.config_manager.get_available_models()
        
        # 构建已加载模型列表
        loaded_models = []
        for model_key in self.models.keys():
            if model_key.startswith('yolo_'):
                model_id = model_key.replace('yolo_', '')
                if model_id in available_models:
                    model_info = available_models[model_id]
                    loaded_models.append({
                        "id": model_id,
                        "name": model_info["config"]["name"],
                        "type": model_info["config"]["type"],
                        "status": "loaded",
                        "file_size": model_info.get("file_size", 0)
                    })
            else:
                loaded_models.append({
                    "id": model_key,
                    "name": model_key,
                    "type": "builtin",
                    "status": "loaded"
                })
        
        info = {
            "device": self.device,
            "loaded_models": loaded_models,
            "local_models_summary": models_summary,
            "cuda_available": torch.cuda.is_available(),
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "default_model": self.config_manager.get_default_model()
        }
        
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory
        
        return {"success": True, "info": info}
    
    def reload_models(self) -> Dict[str, Any]:
        """重新加载模型"""
        try:
            # 清空现有模型
            self.models = {}
            
            # 重新初始化
            self._init_models()
            
            return {
                "success": True,
                "message": "模型重新加载成功",
                "loaded_count": len(self.models)
            }
        except Exception as e:
            logger.error(f"重新加载模型失败: {e}")
            return {"success": False, "error": str(e)}
    
    def get_available_model_list(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        try:
            available_models = self.config_manager.get_available_models()
            models_list = []
            
            for model_id, model_info in available_models.items():
                models_list.append({
                    "id": model_id,
                    "name": model_info["config"]["name"],
                    "description": model_info["config"]["description"],
                    "type": model_info["config"]["type"],
                    "task": model_info["config"]["task"],
                    "status": model_info["status"],
                    "file_size": model_info.get("file_size", 0),
                    "file_path": model_info["file_path"]
                })
            
            return {
                "success": True,
                "models": models_list,
                "total_count": len(models_list)
            }
        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return {"success": False, "error": str(e)}

# 全局模型管理器实例
local_model_manager = LocalModelManager()
