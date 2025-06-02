from fastapi import APIRouter, HTTPException, Depends
import json
from pathlib import Path
from typing import List
from lightrag.api.utils_api import get_combined_auth_dependency
from typing import Optional

router = APIRouter(tags=["courses"])

# 獲取 courses.json 的路徑
COURSES_JSON_PATH = Path(__file__).parent.parent.parent / "data" / "courses.json"

def create_course_routes(api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/courses", response_model=List[str], dependencies=[Depends(combined_auth)])
    async def get_courses():
        """
        獲取所有課程 ID
        """
        try:
            if not COURSES_JSON_PATH.exists():
                return []
                
            with open(COURSES_JSON_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get("courseIds", [])
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"讀取課程列表失敗: {str(e)}") 
    
    return router 