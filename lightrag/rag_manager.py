import asyncio
from typing import Dict, Optional, List
from lightrag import LightRAG
import logging

logger = logging.getLogger(__name__)

class RAGManager:
    """管理多個 LightRAG 實例的類別"""
    
    def __init__(
        self,
        max_preloaded: int = 10,
        working_dir: str = None,
        llm_model_func = None,
        llm_model_name: str = None,
        llm_model_max_async: int = None,
        llm_model_max_token_size: int = None,
        chunk_token_size: int = None,
        chunk_overlap_token_size: int = None,
        llm_model_kwargs: dict = None,
        embedding_func = None,
        kv_storage: str = None,
        graph_storage: str = None,
        vector_storage: str = None,
        doc_status_storage: str = None,
        vector_db_storage_cls_kwargs: dict = None,
        enable_llm_cache_for_entity_extract: bool = None,
        embedding_cache_config: dict = None,
        namespace_prefix: str = None,
        auto_manage_storages_states: bool = False,
        max_parallel_insert: int = None,
        addon_params: dict = None
    ):
        """
        初始化 RAGManager
        
        Args:
            max_preloaded: 最大預載入的課程數量
            working_dir: 工作目錄
            llm_model_func: LLM 模型函數
            llm_model_name: LLM 模型名稱
            llm_model_max_async: LLM 最大並行數
            llm_model_max_token_size: LLM 最大 token 數
            chunk_token_size: 分塊大小
            chunk_overlap_token_size: 分塊重疊大小
            llm_model_kwargs: LLM 模型參數
            embedding_func: 嵌入函數
            kv_storage: KV 儲存類型
            graph_storage: 圖形儲存類型
            vector_storage: 向量儲存類型
            doc_status_storage: 文件狀態儲存類型
            vector_db_storage_cls_kwargs: 向量資料庫儲存類別參數
            enable_llm_cache_for_entity_extract: 是否啟用實體提取的 LLM 快取
            embedding_cache_config: 嵌入快取配置
            namespace_prefix: 命名空間前綴
            auto_manage_storages_states: 是否自動管理儲存狀態
            max_parallel_insert: 最大並行插入數
            addon_params: 附加參數
        """
        self.rag_instances: Dict[str, LightRAG] = {}
        self.max_preloaded = max_preloaded
        self.course_access_count: Dict[str, int] = {}
        self._initialization_lock = asyncio.Lock()
        
        # 儲存 LightRAG 初始化參數
        self.rag_kwargs = {
            "working_dir": working_dir,
            "llm_model_func": llm_model_func,
            "llm_model_name": llm_model_name,
            "llm_model_max_async": llm_model_max_async,
            "llm_model_max_token_size": llm_model_max_token_size,
            "chunk_token_size": chunk_token_size,
            "chunk_overlap_token_size": chunk_overlap_token_size,
            "llm_model_kwargs": llm_model_kwargs,
            "embedding_func": embedding_func,
            "kv_storage": kv_storage,
            "graph_storage": graph_storage,
            "vector_storage": vector_storage,
            "doc_status_storage": doc_status_storage,
            "vector_db_storage_cls_kwargs": vector_db_storage_cls_kwargs,
            "enable_llm_cache_for_entity_extract": enable_llm_cache_for_entity_extract,
            "embedding_cache_config": embedding_cache_config,
            "namespace_prefix": namespace_prefix,
            "auto_manage_storages_states": auto_manage_storages_states,
            "max_parallel_insert": max_parallel_insert,
            "addon_params": addon_params
        }
    
    async def initialize_course(self, course_id: str, **kwargs) -> LightRAG:
        """
        初始化特定課程的 LightRAG 實例
        
        Args:
            course_id: 課程ID
            **kwargs: 傳遞給 LightRAG 的其他參數（會覆蓋預設參數）
            
        Returns:
            LightRAG: 初始化後的實例
        """
        async with self._initialization_lock:
            if course_id in self.rag_instances:
                return self.rag_instances[course_id]
            
            # 如果超過預載入限制，移除最不常用的實例
            if len(self.rag_instances) >= self.max_preloaded:
                least_used = min(self.course_access_count.items(), key=lambda x: x[1])[0]
                await self.rag_instances[least_used].finalize_storages()
                del self.rag_instances[least_used]
                del self.course_access_count[least_used]
            
            # 合併預設參數和傳入的參數
            course_kwargs = self.rag_kwargs.copy()
            course_kwargs.update(kwargs)
            
            # 設置課程特定的命名空間前綴
            course_kwargs["namespace_prefix"] = f"course_{course_id}"
            
            # 初始化新的實例
            rag = LightRAG(**course_kwargs)
            await rag.initialize_storages()
            self.rag_instances[course_id] = rag
            self.course_access_count[course_id] = 0
            
            logger.info(f"已初始化課程 {course_id} 的 LightRAG 實例")
            return rag
    
    async def preload_courses(self, course_ids: List[str], **kwargs):
        """
        預載入多個課程
        
        Args:
            course_ids: 要預載入的課程ID列表
            **kwargs: 傳遞給 LightRAG 的其他參數
        """
        tasks = []
        for course_id in course_ids[:self.max_preloaded]:
            tasks.append(self.initialize_course(course_id, **kwargs))
        
        await asyncio.gather(*tasks)
        logger.info(f"已預載入 {len(tasks)} 個課程")
    
    async def get_rag(self, course_id: str) -> LightRAG:
        """
        獲取特定課程的 LightRAG 實例
        
        Args:
            course_id: 課程ID
            
        Returns:
            LightRAG: 對應的實例
        """
        if course_id not in self.rag_instances:
            await self.initialize_course(course_id)
        
        # 更新訪問計數
        self.course_access_count[course_id] = self.course_access_count.get(course_id, 0) + 1
        return self.rag_instances[course_id]
    
    async def switch_course(self, course_id: str) -> bool:
        """
        切換到特定課程
        
        Args:
            course_id: 要切換到的課程ID
            
        Returns:
            bool: 是否切換成功
        """
        try:
            await self.get_rag(course_id)
            return True
        except Exception as e:
            logger.error(f"切換課程 {course_id} 失敗: {e}")
            return False
    
    async def cleanup(self):
        """清理所有實例"""
        for course_id, rag in self.rag_instances.items():
            try:
                await rag.finalize_storages()
                logger.info(f"已清理課程 {course_id} 的實例")
            except Exception as e:
                logger.error(f"清理課程 {course_id} 的實例失敗: {e}")
        
        self.rag_instances.clear()
        self.course_access_count.clear() 