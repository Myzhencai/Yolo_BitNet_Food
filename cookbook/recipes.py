import httpx
from typing import List
from ..types.models import Recipe
import os
import json

# 远程菜谱JSON文件URL
# RECIPES_URL = 'https://mp-bc8d1f0a-3356-4a4e-8592-f73a3371baa2.cdn.bspapp.com/all_recipes.json'
RECIPES_URL = 'D:/Yolo_BitNet_Food/howtocook-py-mcp/all_recipes.json'

# 从远程URL或本地文件获取数据的异步函数
async def fetch_recipes() -> List[Recipe]:
    try:
        if RECIPES_URL.startswith('http://') or RECIPES_URL.startswith('https://'):
            # 使用httpx异步获取远程数据
            async with httpx.AsyncClient() as client:
                response = await client.get(RECIPES_URL)
                if response.status_code != 200:
                    raise Exception(f"HTTP error! Status: {response.status_code}")
                data = response.json()
        else:
            # 读取本地文件
            if not os.path.exists(RECIPES_URL):
                raise Exception(f"本地文件不存在: {RECIPES_URL}")
            with open(RECIPES_URL, 'r', encoding='utf-8') as f:
                data = json.load(f)
        return [Recipe.model_validate(recipe) for recipe in data]
    except Exception as error:
        print(f'获取菜谱数据失败: {error}')
        # 直接返回空列表
        return []

# 获取所有分类
def get_all_categories(recipes: List[Recipe]) -> List[str]:
    categories = set()
    for recipe in recipes:
        if recipe.category:
            categories.add(recipe.category)
    return list(categories) 