import uuid
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma  # pip install langchain-chroma
from src.config import DB_DIR

class ToolMemory:
    def __init__(self, persist_dir=DB_DIR):
        self.embeddings = OpenAIEmbeddings()
        
        self.vector_store = Chroma(
            collection_name="math_tools",
            embedding_function=self.embeddings,
            persist_directory=persist_dir
        )

    def add_tool(self, tool_info: Dict[str, str]):
        """
        [저장]
        - page_content: docstring (검색 대상)
        - metadata: code, name (실제 사용할 정보)
        """
        doc = Document(
            page_content=tool_info['docstring'], # docstring embedding
            metadata={
                "tool_name": tool_info['name'],
                "tool_code": tool_info['code'],
                "type": "function"
            },
            id=str(uuid.uuid4())
        )
        self.vector_store.add_documents([doc])
        # Chroma는 자동 저장되지만, 명시적 저장이 필요하면: self.vector_store.persist()

    def search_tools(self, query: str, k: int = 5) -> List[Dict]:
        """
        [검색]
        Subtask Description(query)과 유사한 Docstring을 가진 도구들을 찾습니다.
        """
        results = self.vector_store.similarity_search_with_score(query, k=k)
        
        candidates = []
        for doc, score in results:
            # Score filtering (L2 distance: lower is better / Cosine: higher is better)
            # 여기서는 편의상 모든 결과를 반환하고 LLM이 판단하게 함
            candidates.append({
                "name": doc.metadata.get("tool_name"),
                "docstring": doc.page_content,
                "code": doc.metadata.get("tool_code"),
                "score": score
            })
            
        return candidates