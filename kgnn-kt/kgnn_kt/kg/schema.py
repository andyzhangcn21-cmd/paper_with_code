from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Literal, Optional

NodeType = Literal["problem", "concept"]
EdgeType = Literal["requires", "subclass_of", "antonym"]

class Node(BaseModel):
    node_id: str
    node_type: NodeType
    label: str
    attrs: Dict[str, str] = Field(default_factory=dict)

class Edge(BaseModel):
    src: str
    dst: str
    edge_type: EdgeType
    attrs: Dict[str, str] = Field(default_factory=dict)

class Graph(BaseModel):
    nodes: List[Node] = Field(default_factory=list)
    edges: List[Edge] = Field(default_factory=list)

    def node_index(self) -> Dict[str, int]:
        return {n.node_id: i for i, n in enumerate(self.nodes)}
