import pandas as pd
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from src.config import DB_DIR

load_dotenv()

def inspect_tool_db():
    print(f"📂 Loading Vector DB")
    
    print(DB_DIR)
    # 1. DB 로드 (기존에 저장된 경로 지정)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma(
        collection_name="math_tools", # memory.py에서 쓴 이름과 같아야 함
        embedding_function=embeddings,
        persist_directory=DB_DIR
    )

    # 2. 모든 데이터 가져오기
    # .get() 메서드는 저장된 모든 문서의 ID, 메타데이터, 내용을 반환합니다.
    all_data = vector_store.get()
    
    total_count = len(all_data['ids'])
    print(f"📊 Total Tools Stored: {total_count}\n")
    
    if total_count == 0:
        print("⚠️ No tools found.")
        return

    # 3. 보기 좋게 출력 (Pandas 활용)
    # Chroma는 metadata에 'tool_code', 'tool_name'을 저장했습니다.
    tools_list = []
    for i in range(total_count):
        meta = all_data['metadatas'][i]
        content = all_data['documents'][i]
        
        tools_list.append({
            "ID": all_data['ids'][i],
            "Name": meta.get("tool_name", "N/A"),
            "Docstring (Search Key)": content[:50] + "...", # 너무 기니까 자름
            "Full Code": meta.get("tool_code", "N/A")[:100] + "..." # 코드도 앞부분만
        })


        if "extract" in meta.get("tool_name").lower():
            print(meta.get("tool_code"))

    df = pd.DataFrame(tools_list)
    
    # 터미널에 표 형태로 출력
    print(df.to_markdown(index=False))
    
    # (선택) 전체 코드를 보고 싶으면 특정 ID로 조회 가능
    # print("\n--- Example Code of First Tool ---")
    # print(all_data['metadatas'][-2]['tool_code'])
    print(all_data['metadatas'][-1]['tool_code'])

    

if __name__ == "__main__":
    inspect_tool_db()