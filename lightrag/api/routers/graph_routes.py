"""
This module contains all graph-related routes for the LightRAG API.
"""

from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, Query, HTTPException
from pydantic import BaseModel, Field
import traceback

from ..utils_api import get_combined_auth_dependency
from lightrag.rag_manager import RAGManager

router = APIRouter(tags=["graph"])


def create_graph_routes(rag_manager: RAGManager, api_key: Optional[str] = None):
    combined_auth = get_combined_auth_dependency(api_key)

    @router.get("/graph/label/list", dependencies=[Depends(combined_auth)])
    async def get_graph_labels(
        course_id: str = Query("", description="課程ID")
    ):
        """
        Get all graph labels

        Returns:
            List[str]: List of graph labels
        """
        rag = await rag_manager.get_rag(course_id)
        return await rag.get_graph_labels()

    @router.get("/graphs", dependencies=[Depends(combined_auth)])
    async def get_knowledge_graph(
        course_id: str = Query("", description="課程ID"),
        label: str = Query(..., description="Label to get knowledge graph for"),
        max_depth: int = Query(3, description="Maximum depth of graph", ge=1),
        max_nodes: int = Query(1000, description="Maximum nodes to return", ge=1),
    ):
        """
        Retrieve a connected subgraph of nodes where the label includes the specified label.
        When reducing the number of nodes, the prioritization criteria are as follows:
            1. Hops(path) to the staring node take precedence
            2. Followed by the degree of the nodes

        Args:
            course_id (str): The course ID for the query.
            label (str): Label of the starting node
            max_depth (int, optional): Maximum depth of the subgraph,Defaults to 3
            max_nodes: Maxiumu nodes to return

        Returns:
            Dict[str, List[str]]: Knowledge graph for label
        """
        rag = await rag_manager.get_rag(course_id)
        return await rag.get_knowledge_graph(
            node_label=label,
            max_depth=max_depth,
            max_nodes=max_nodes,
        )

    @router.get("/graph/node/neighbors", dependencies=[Depends(combined_auth)])
    async def get_node_neighbors(
        course_id: str = Query("", description="課程ID"),
        node_id: str = Query(..., description="Node ID to get neighbors for")
    ):
        """
        獲取指定節點的相鄰節點和邊

        Args:
            course_id (str): The course ID for the query.
            node_id (str): 目標節點ID
            max_depth (int, optional): 最大深度,預設為1
            max_nodes (int, optional): 最大回傳節點數,預設為100

        Returns:
            Dict: 包含相鄰節點和邊的子圖
        """
        rag = await rag_manager.get_rag(course_id)
        return await rag.get_node_edges(
            node_id=node_id
        )

    class SourceGraphResponse(BaseModel):
        """Response model for source document graph data

        Attributes:
            nodes: List of nodes in the graph
            relationships: List of relationships in the graph
        """
        nodes: List[Dict[str, Any]] = Field(description="List of nodes in the graph")
        relationships: List[Dict[str, Any]] = Field(description="List of relationships in the graph")

    @router.get(
        "/source/{source_id}",
        response_model=SourceGraphResponse,
        dependencies=[Depends(combined_auth)]
    )
    async def get_source_graph(
        source_id: str,
        course_id: str = Query("", description="課程ID")
    ):
        """
        Get all nodes and relationships for a specific source document.

        This endpoint retrieves all nodes and relationships that were extracted from
        a specific source document identified by its ID.

        Args:
            source_id (str): The ID of the source document
            course_id (str): The course ID for organizing data

        Returns:
            SourceGraphResponse: A response object containing:
                - nodes: List of nodes extracted from the source
                - relationships: List of relationships between these nodes

        Raises:
            HTTPException: If the source document is not found (404) or other errors occur (500)
        """
        try:
            rag = await rag_manager.get_rag(course_id)
            
            # 通過 rag.kg 訪問 Neo4j 存儲實例
            nodes, relationships = await rag.chunk_entity_relation_graph.get_nodes_and_relationships_by_file_path(source_id)
            
            if not nodes:
                raise HTTPException(
                    status_code=404,
                    detail=f"No nodes found for source document {source_id}"
                )

            return SourceGraphResponse(
                nodes=nodes,
                relationships=relationships
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return router