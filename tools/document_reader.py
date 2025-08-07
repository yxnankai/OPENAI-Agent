#!/usr/bin/env python3
"""
文档读取工具 - 支持txt和doc文件
"""
import os
import re
import logging
from typing import List, Dict, Optional
import subprocess
import tempfile

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentReader:
    """文档读取器"""
    
    def __init__(self):
        self.supported_extensions = ['.txt', '.doc', '.docx']
    
    def search_files(self, search_path: str, query: str) -> List[Dict]:
        """
        搜索文件
        Args:
            search_path: 搜索路径
            query: 搜索关键词
        Returns:
            匹配的文件列表
        """
        matching_files = []
        
        # 检查搜索路径是否存在
        if not os.path.exists(search_path):
            logger.error(f"搜索路径不存在: {search_path}")
            return matching_files
        
        if not os.path.isdir(search_path):
            logger.error(f"搜索路径不是目录: {search_path}")
            return matching_files
        
        try:
            logger.info(f"开始在 {search_path} 中搜索包含 '{query}' 的文档文件")
            
            # 使用Windows的dir命令搜索文件
            cmd = f'dir /s /b "{search_path}\\*{query}*"'
            logger.info(f"执行命令: {cmd}")
            
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                files = result.stdout.strip().split('\n')
                logger.info(f"dir命令找到 {len(files)} 个文件")
                
                for file_path in files:
                    if file_path and os.path.exists(file_path):
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext in self.supported_extensions:
                            try:
                                file_info = {
                                    'path': file_path,
                                    'name': os.path.basename(file_path),
                                    'size': os.path.getsize(file_path),
                                    'extension': file_ext
                                }
                                matching_files.append(file_info)
                                logger.info(f"找到文档文件: {file_path}")
                            except OSError as e:
                                logger.warning(f"无法获取文件信息 {file_path}: {e}")
            
            # 如果dir命令失败或没有找到文件，使用PowerShell
            if not matching_files:
                logger.info("dir命令未找到文件，尝试使用PowerShell")
                ps_cmd = f'Get-ChildItem -Path "{search_path}" -Recurse -Name "*{query}*" | Where-Object {{ $_ -match "\\.(txt|doc|docx)$" }}'
                result = subprocess.run(['powershell', '-Command', ps_cmd], 
                                      capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    files = result.stdout.strip().split('\n')
                    logger.info(f"PowerShell找到 {len(files)} 个文件")
                    
                    for file_name in files:
                        if file_name:
                            file_path = os.path.join(search_path, file_name)
                            if os.path.exists(file_path):
                                file_ext = os.path.splitext(file_path)[1].lower()
                                try:
                                    file_info = {
                                        'path': file_path,
                                        'name': os.path.basename(file_path),
                                        'size': os.path.getsize(file_path),
                                        'extension': file_ext
                                    }
                                    matching_files.append(file_info)
                                    logger.info(f"PowerShell找到文档文件: {file_path}")
                                except OSError as e:
                                    logger.warning(f"无法获取文件信息 {file_path}: {e}")
                else:
                    logger.warning(f"PowerShell命令失败: {result.stderr}")
                                
        except subprocess.TimeoutExpired:
            logger.error(f"搜索超时: {search_path}")
        except Exception as e:
            logger.error(f"搜索文件时出错: {e}")
        
        logger.info(f"总共找到 {len(matching_files)} 个匹配的文档文件")
        return matching_files
    
    def read_txt_file(self, file_path: str) -> str:
        """读取txt文件"""
        try:
            logger.info(f"尝试读取txt文件: {file_path}")
            # 尝试UTF-8编码
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                logger.info(f"成功读取txt文件，长度: {len(content)} 字符")
                return content
        except UnicodeDecodeError:
            logger.info("UTF-8编码失败，尝试GBK编码")
            try:
                with open(file_path, 'r', encoding='gbk') as f:
                    content = f.read()
                    logger.info(f"GBK编码成功，长度: {len(content)} 字符")
                    return content
            except UnicodeDecodeError:
                logger.info("GBK编码失败，尝试latin-1编码")
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                        logger.info(f"latin-1编码成功，长度: {len(content)} 字符")
                        return content
                except Exception as e:
                    logger.error(f"所有编码都失败: {e}")
                    return f"❌ 无法读取文件，编码问题: {file_path}"
        except Exception as e:
            logger.error(f"读取txt文件失败: {e}")
            return f"❌ 读取txt文件失败: {str(e)}"
    
    def read_doc_file(self, file_path: str) -> str:
        """读取doc文件"""
        try:
            logger.info(f"尝试读取doc文件: {file_path}")
            # 使用antiword工具读取doc文件
            result = subprocess.run(['antiword', file_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                content = result.stdout
                logger.info(f"成功读取doc文件，长度: {len(content)} 字符")
                return content
            else:
                logger.error(f"antiword命令失败: {result.stderr}")
                return f"❌ 无法读取doc文件: {file_path}，错误: {result.stderr}"
        except FileNotFoundError:
            logger.error("未找到antiword工具")
            return f"❌ 未找到antiword工具，无法读取doc文件: {file_path}"
        except Exception as e:
            logger.error(f"读取doc文件时出错: {e}")
            return f"❌ 读取doc文件时出错: {str(e)}"
    
    def read_docx_file(self, file_path: str) -> str:
        """读取docx文件"""
        try:
            logger.info(f"尝试读取docx文件: {file_path}")
            # 使用python-docx库读取docx文件
            from docx import Document
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            content = '\n'.join(text)
            logger.info(f"成功读取docx文件，长度: {len(content)} 字符")
            return content
        except ImportError:
            logger.error("未安装python-docx库")
            return f"❌ 未安装python-docx库，无法读取docx文件: {file_path}"
        except Exception as e:
            logger.error(f"读取docx文件时出错: {e}")
            return f"❌ 读取docx文件时出错: {str(e)}"
    
    def read_document(self, file_path: str) -> str:
        """读取文档内容"""
        if not os.path.exists(file_path):
            logger.error(f"文件不存在: {file_path}")
            return f"❌ 文件不存在: {file_path}"
        
        if not os.path.isfile(file_path):
            logger.error(f"路径不是文件: {file_path}")
            return f"❌ 路径不是文件: {file_path}"
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext == '.txt':
            return self.read_txt_file(file_path)
        elif file_ext == '.doc':
            return self.read_doc_file(file_path)
        elif file_ext == '.docx':
            return self.read_docx_file(file_path)
        else:
            logger.error(f"不支持的文件格式: {file_ext}")
            return f"❌ 不支持的文件格式: {file_ext}，支持的格式: {', '.join(self.supported_extensions)}"
    
    def extract_summary(self, content: str, max_length: int = 1000) -> str:
        """提取文档摘要"""
        if len(content) <= max_length:
            return content
        
        # 简单的摘要提取：取前500字符和后500字符
        first_part = content[:500]
        last_part = content[-500:]
        
        # 找到第一个句号的位置
        first_sentence_end = first_part.find('。')
        if first_sentence_end > 0:
            first_part = first_part[:first_sentence_end + 1]
        
        # 找到最后一个句号的位置
        last_sentence_start = last_part.rfind('。')
        if last_sentence_start > 0:
            last_part = last_part[last_sentence_start + 1:]
        
        return f"{first_part}...\n\n[内容省略]\n\n...{last_part}"
    
    def search_and_read(self, search_path: str, query: str, max_files: int = 5) -> Dict:
        """
        搜索并读取文档
        Args:
            search_path: 搜索路径
            query: 搜索关键词
            max_files: 最大读取文件数
        Returns:
            搜索结果和内容
        """
        logger.info(f"开始搜索并读取文档: 路径={search_path}, 关键词={query}, 最大文件数={max_files}")
        
        # 搜索文件
        matching_files = self.search_files(search_path, query)
        
        if not matching_files:
            logger.info(f"未找到包含 '{query}' 的文档文件")
            return {
                'success': False,
                'message': f'在 {search_path} 中未找到包含 "{query}" 的文档文件',
                'files': [],
                'contents': []
            }
        
        # 限制文件数量
        matching_files = matching_files[:max_files]
        logger.info(f"将读取 {len(matching_files)} 个文件")
        
        # 读取文件内容
        contents = []
        for file_info in matching_files:
            try:
                logger.info(f"读取文件: {file_info['path']}")
                content = self.read_document(file_info['path'])
                summary = self.extract_summary(content)
                
                file_content = {
                    'file_info': file_info,
                    'full_content': content,
                    'summary': summary,
                    'content_length': len(content)
                }
                contents.append(file_content)
                logger.info(f"成功读取文件: {file_info['name']}")
                
            except Exception as e:
                logger.error(f"读取文件失败 {file_info['path']}: {e}")
                file_content = {
                    'file_info': file_info,
                    'error': f'读取文件时出错: {str(e)}',
                    'full_content': '',
                    'summary': '',
                    'content_length': 0
                }
                contents.append(file_content)
        
        logger.info(f"搜索并读取完成，成功读取 {len([c for c in contents if 'error' not in c])} 个文件")
        return {
            'success': True,
            'message': f'找到 {len(matching_files)} 个相关文档',
            'files': matching_files,
            'contents': contents
        } 