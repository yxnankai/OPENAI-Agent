#!/usr/bin/env python3
"""
增强的网页搜索工具
能够获取搜索结果并使用AI进行总结
"""
import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import List, Dict, Any
import time
import random

class EnhancedWebSearch:
    """增强的网页搜索工具"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
    
    def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """使用DuckDuckGo搜索"""
        try:
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # DuckDuckGo搜索结果选择器
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results[:max_results]:
                try:
                    # 提取标题
                    title_elem = result.find('a', class_='result__a')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    
                    # 提取摘要
                    snippet_elem = result.find('a', class_='result__snippet')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    # 提取更多内容
                    content_elem = result.find('div', class_='result__body')
                    content = content_elem.get_text(strip=True) if content_elem else ""
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'content': content,
                            'url': url,
                            'source': 'DuckDuckGo'
                        })
                        
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            print(f"DuckDuckGo搜索失败: {e}")
            return []
    
    def search_bing(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """使用Bing搜索"""
        try:
            search_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            search_results = soup.find_all('li', class_='b_algo')
            
            for result in search_results[:max_results]:
                try:
                    title_elem = result.find('h2')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    snippet_elem = result.find('p')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'content': snippet,
                            'url': '',
                            'source': 'Bing'
                        })
                        
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            print(f"Bing搜索失败: {e}")
            return []
    
    def search_baidu(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """使用百度搜索"""
        try:
            search_url = f"https://www.baidu.com/s?wd={urllib.parse.quote(query)}"
            response = self.session.get(search_url, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            search_results = soup.find_all('div', class_='result')
            
            for result in search_results[:max_results]:
                try:
                    title_elem = result.find('h3')
                    if not title_elem:
                        continue
                    
                    title = title_elem.get_text(strip=True)
                    
                    snippet_elem = result.find('div', class_='c-abstract')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    if title and snippet:
                        results.append({
                            'title': title,
                            'snippet': snippet,
                            'content': snippet,
                            'url': '',
                            'source': 'Baidu'
                        })
                        
                except Exception:
                    continue
            
            return results
            
        except Exception as e:
            print(f"百度搜索失败: {e}")
            return []
    
    def search_multiple_sources(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """从多个搜索引擎获取结果"""
        all_results = []
        
        # 尝试多个搜索引擎
        search_methods = [
            self.search_duckduckgo,
            self.search_bing,
            self.search_baidu
        ]
        
        for search_method in search_methods:
            try:
                results = search_method(query, max_results)
                all_results.extend(results)
                
                # 如果获取到足够的结果，就停止
                if len(all_results) >= max_results * 2:
                    break
                    
                # 添加延迟避免被封
                time.sleep(random.uniform(1, 3))
                
            except Exception as e:
                print(f"搜索方法 {search_method.__name__} 失败: {e}")
                continue
        
        # 去重并限制结果数量
        unique_results = []
        seen_titles = set()
        
        for result in all_results:
            if result['title'] not in seen_titles:
                unique_results.append(result)
                seen_titles.add(result['title'])
                
                if len(unique_results) >= max_results:
                    break
        
        return unique_results
    
    def format_search_results(self, results: List[Dict[str, str]], query: str) -> str:
        """格式化搜索结果"""
        if not results:
            return f"🌐 搜索 '{query}' 的结果:\n\n暂时无法获取搜索结果。建议您：\n1. 检查网络连接\n2. 尝试其他搜索关键词\n3. 直接访问搜索引擎网站"
        
        result_text = f"🌐 搜索 '{query}' 的结果:\n\n"
        result_text += f"📊 查询关键词: {query}\n"
        result_text += f"📋 找到 {len(results)} 个相关结果\n\n"
        
        # 资料源列表
        result_text += "📚 参考资料源:\n"
        result_text += "=" * 50 + "\n\n"
        
        for i, result in enumerate(results, 1):
            result_text += f"📖 资料源 {i}:\n"
            result_text += f"   标题: {result['title']}\n"
            result_text += f"   来源: {result['source']}\n"
            if result['url']:
                result_text += f"   链接: {result['url']}\n"
            result_text += f"   摘要: {result['snippet']}\n"
            if result['content'] and result['content'] != result['snippet']:
                result_text += f"   详细内容: {result['content'][:300]}...\n"
            result_text += "\n"
        
        # 总结部分
        result_text += "📈 搜索结果总结:\n"
        result_text += "=" * 50 + "\n"
        
        # 提取关键信息进行总结
        summary = self._generate_summary(results, query)
        result_text += summary
        
        return result_text
    
    def _generate_summary(self, results: List[Dict[str, str]], query: str) -> str:
        """生成搜索结果总结"""
        if not results:
            return "暂无相关信息可总结。"
        
        summary = f"🔍 关于 '{query}' 的信息总结:\n\n"
        
        # 主要信息来源统计
        sources = {}
        for result in results:
            source = result['source']
            sources[source] = sources.get(source, 0) + 1
        
        summary += "📊 信息来源分布:\n"
        for source, count in sources.items():
            summary += f"   • {source}: {count} 条信息\n"
        summary += "\n"
        
        # 关键信息提取
        summary += "📋 关键信息摘要:\n"
        key_points = []
        
        for i, result in enumerate(results[:3], 1):  # 取前3个最重要的结果
            title = result['title']
            snippet = result['snippet']
            
            # 提取关键数据
            if any(keyword in snippet.lower() for keyword in ['数据', '统计', '报告', '分析', '预测']):
                key_points.append(f"{i}. {title} - {snippet[:100]}...")
        
        if key_points:
            for point in key_points:
                summary += f"   {point}\n"
        else:
            summary += "   暂无具体数据信息\n"
        
        summary += "\n💡 建议:\n"
        summary += "   • 建议查看官方统计数据获取准确信息\n"
        summary += "   • 关注权威机构发布的分析报告\n"
        summary += "   • 对比多个来源的信息以确保准确性\n"
        summary += "   • 如需更详细信息，可直接访问上述链接\n"
        
        return summary
    
    def search_and_format(self, query: str, max_results: int = 5) -> str:
        """搜索并格式化结果"""
        results = self.search_multiple_sources(query, max_results)
        return self.format_search_results(results, query)

# 全局实例
enhanced_search = EnhancedWebSearch() 