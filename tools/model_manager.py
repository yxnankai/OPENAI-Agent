import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class ModelConfigManager:
    """模型配置管理器"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.config_file = self.models_dir / "config.json"
        self.models_config = {}
        self.available_models = {}
        
        # 确保目录存在
        self.models_dir.mkdir(exist_ok=True)
        
        # 加载配置
        self._load_config()
        self._scan_models()
    
    def _load_config(self):
        """加载模型配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.models_config = json.load(f)
                logger.info(f"成功加载模型配置: {self.config_file}")
            else:
                # 创建默认配置
                self._create_default_config()
        except Exception as e:
            logger.error(f"加载模型配置失败: {e}")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        default_config = {
            "models": {
                "yolov8n": {
                    "name": "YOLOv8 Nano",
                    "type": "yolov8",
                    "description": "YOLOv8 轻量级目标检测模型",
                    "file": "yolov8n.pt",
                    "task": "detection",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                },
                "yolov8s": {
                    "name": "YOLOv8 Small",
                    "type": "yolov8",
                    "description": "YOLOv8 小型目标检测模型",
                    "file": "yolov8s.pt",
                    "task": "detection",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                },
                "yolov8m": {
                    "name": "YOLOv8 Medium",
                    "type": "yolov8",
                    "description": "YOLOv8 中型目标检测模型",
                    "file": "yolov8m.pt",
                    "task": "detection",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                },
                "yolov8l": {
                    "name": "YOLOv8 Large",
                    "type": "yolov8",
                    "description": "YOLOv8 大型目标检测模型",
                    "file": "yolov8l.pt",
                    "task": "detection",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                },
                "yolov8x": {
                    "name": "YOLOv8 XLarge",
                    "type": "yolov8",
                    "description": "YOLOv8 超大型目标检测模型",
                    "file": "yolov8x.pt",
                    "task": "detection",
                    "input_size": [640, 640],
                    "confidence_threshold": 0.5,
                    "iou_threshold": 0.45
                }
            },
            "settings": {
                "default_model": "yolov8n",
                "auto_download": True,
                "cache_dir": "models/cache"
            }
        }
        
        self.models_config = default_config
        self._save_config()
        logger.info("创建默认模型配置")
    
    def _save_config(self):
        """保存配置到文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.models_config, f, indent=2, ensure_ascii=False)
            logger.info(f"配置已保存到: {self.config_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def _scan_models(self):
        """扫描可用的模型文件"""
        try:
            available_models = {}
            
            for model_id, config in self.models_config.get("models", {}).items():
                model_file = self.models_dir / config.get("file", "")
                
                if model_file.exists():
                    available_models[model_id] = {
                        "config": config,
                        "file_path": str(model_file),
                        "file_size": model_file.stat().st_size,
                        "status": "available"
                    }
                else:
                    available_models[model_id] = {
                        "config": config,
                        "file_path": str(model_file),
                        "status": "missing"
                    }
            
            self.available_models = available_models
            logger.info(f"扫描到 {len(available_models)} 个模型配置")
            
        except Exception as e:
            logger.error(f"扫描模型失败: {e}")
    
    def get_model_config(self, model_id: str) -> Optional[Dict]:
        """获取模型配置"""
        return self.models_config.get("models", {}).get(model_id)
    
    def get_available_models(self) -> Dict[str, Any]:
        """获取可用模型列表"""
        return self.available_models
    
    def get_model_path(self, model_id: str) -> Optional[str]:
        """获取模型文件路径"""
        model_info = self.available_models.get(model_id)
        if model_info and model_info["status"] == "available":
            return model_info["file_path"]
        return None
    
    def is_model_available(self, model_id: str) -> bool:
        """检查模型是否可用"""
        model_info = self.available_models.get(model_id)
        return model_info is not None and model_info["status"] == "available"
    
    def get_default_model(self) -> str:
        """获取默认模型ID"""
        return self.models_config.get("settings", {}).get("default_model", "yolov8n")
    
    def add_model(self, model_id: str, config: Dict):
        """添加新模型配置"""
        try:
            self.models_config["models"][model_id] = config
            self._save_config()
            self._scan_models()
            logger.info(f"添加模型配置: {model_id}")
            return True
        except Exception as e:
            logger.error(f"添加模型配置失败: {e}")
            return False
    
    def remove_model(self, model_id: str) -> bool:
        """移除模型配置"""
        try:
            if model_id in self.models_config["models"]:
                del self.models_config["models"][model_id]
                self._save_config()
                self._scan_models()
                logger.info(f"移除模型配置: {model_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"移除模型配置失败: {e}")
            return False
    
    def update_model_config(self, model_id: str, config: Dict) -> bool:
        """更新模型配置"""
        try:
            if model_id in self.models_config["models"]:
                self.models_config["models"][model_id].update(config)
                self._save_config()
                self._scan_models()
                logger.info(f"更新模型配置: {model_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"更新模型配置失败: {e}")
            return False
    
    def get_models_summary(self) -> Dict[str, Any]:
        """获取模型摘要信息"""
        summary = {
            "total_models": len(self.available_models),
            "available_count": 0,
            "missing_count": 0,
            "models": {}
        }
        
        for model_id, info in self.available_models.items():
            status = info["status"]
            if status == "available":
                summary["available_count"] += 1
            else:
                summary["missing_count"] += 1
            
            summary["models"][model_id] = {
                "name": info["config"]["name"],
                "type": info["config"]["type"],
                "task": info["config"]["task"],
                "status": status,
                "file_size": info.get("file_size", 0) if status == "available" else 0
            }
        
        return summary

# 全局模型配置管理器实例
model_config_manager = ModelConfigManager()
