import ast
import re
from typing import List, Dict

def parse_tools_from_code(llm_output: str) -> List[Dict[str, str]]:
    """
    LLM 응답 내의 코드 블록을 파싱하여 여러 개의 도구(함수) 리스트로 반환합니다.
    
    [기능]
    1. 코드 상단의 모든 Import 문을 추출합니다 (공통 사용).
    2. 정의된 모든 함수(FunctionDef)를 각각 추출합니다.
    3. 각 함수에 공통 Import 문을 붙여 독립 실행 가능한 코드로 만듭니다.
    4. 각 함수의 Docstring을 추출합니다.
    
    Returns:
        [
            {
                "name": "func_a",
                "code": "import numpy as np\n\ndef func_a()...",
                "docstring": "Calculates A..."
            },
            ...
        ]
    """
    # 1. 마크다운 코드 블록 제거
    clean_code = re.sub(r"```python\n|```", "", llm_output).strip()
    
    try:
        tree = ast.parse(clean_code)
    except SyntaxError as e:
        print(f"❌ Syntax Error in generated code: {e}")
        return []

    global_imports = []
    extracted_tools = []

    # 2. AST 순회
    for node in tree.body:
        # (A) Import 문 수집 (import x, from x import y)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            segment = ast.get_source_segment(clean_code, node)
            if segment:
                global_imports.append(segment)
        
        # (B) 함수 정의 수집
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            segment = ast.get_source_segment(clean_code, node)
            if segment:
                # Docstring 추출
                docstring = ast.get_docstring(node) or f"Tool named {node.name}"
                
                extracted_tools.append({
                    "name": node.name,
                    "body": segment,
                    "docstring": docstring.strip()
                })

    # 3. Import 문과 함수 결합
    import_header = "\n".join(global_imports)
    
    final_tools = []
    for tool in extracted_tools:
        # Import가 있으면 붙이고, 없으면 함수 본문만
        full_code = f"{import_header}\n\n{tool['body']}" if import_header else tool['body']
        
        final_tools.append({
            "name": tool['name'],
            "code": clean_code,      # 💾 실행/저장용 (Import 포함)
            "docstring": tool['docstring'] # 🔍 검색용
        })
        
    return final_tools