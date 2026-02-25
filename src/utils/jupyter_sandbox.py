import json
import queue
import atexit
import logging
from jupyter_client.manager import KernelManager
from ..config import TEST_DIR
import os
import shutil
import tempfile

logger = logging.getLogger(__name__)

class SingleKernel:
    def __init__(self, work_dir="./"):
        self.work_dir = os.path.abspath(work_dir)
        os.makedirs(self.work_dir, exist_ok=True)
        self.connection_dir = None
        self.km = None
        self.kc = None
        self._start_kernel()

    
    def _start_kernel(self):
        """커널을 (재)시작하는 내부 메서드"""
        # 1. 연결 및 런타임 파일용 안전한 임시 디렉토리 생성
        self.connection_dir = tempfile.mkdtemp()
        
        # 🔥 [핵심 1] Jupyter가 런타임 파일(소켓 등)을 
        # 절대 작업 폴더에 만들지 않도록 환경변수 강제 설정
        os.environ["JUPYTER_RUNTIME_DIR"] = self.connection_dir
        
        # 2. KernelManager 설정 (TCP 강제)
        self.km = KernelManager(
            kernel_name='python3',
            connection_dir=self.connection_dir,
            transport='tcp',   # 네트워크 소켓 사용
            ip='127.0.0.1'     # 로컬호스트 강제
        )
        
        # 3. 커널 시작 (with retry)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.km.start_kernel()
                self.kc = self.km.client()
                self.kc.start_channels()
                self.kc.wait_for_ready(timeout=10)
                break  # 성공 시 루프 탈출
            except RuntimeError:
                logger.warning(f"⚠️ Kernel start failed (attempt {attempt+1}/{max_retries}) in {self.work_dir}")
                self.cleanup()
                if attempt < max_retries - 1:
                    import time
                    wait_time = 2 ** (attempt + 1)  # 2, 4, 8초
                    logger.info(f"   Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    # 재시도를 위해 KernelManager 재생성
                    self.connection_dir = tempfile.mkdtemp()
                    os.environ["JUPYTER_RUNTIME_DIR"] = self.connection_dir
                    self.km = KernelManager(
                        kernel_name='python3',
                        connection_dir=self.connection_dir,
                        transport='tcp',
                        ip='127.0.0.1'
                    )
                else:
                    raise RuntimeError(f"Failed to start kernel in {self.work_dir} after {max_retries} attempts")
            except Exception as e:
                self.cleanup()
                raise e
        
        # 4. 워크스페이스로 이동
        # (Jupyter는 런타임 폴더에서 시작했을 수 있으므로 이동 필요)
        self.execute(f"import os; os.chdir('{self.work_dir}')")
    
    def execute(self, code, timeout=30):
        try:
            msg_id = self.kc.execute(code)
        except Exception as e:
            return {"stdout": "", "stderr": str(e)}

        stdout, stderr = [], []
        while True:
            try:
                msg = self.kc.get_iopub_msg(timeout=timeout)
                if msg['parent_header'].get('msg_id') != msg_id: continue
                
                msg_type = msg['msg_type']
                content = msg['content']
                
                if msg_type == 'stream':
                    if content['name'] == 'stdout': stdout.append(content['text'])
                    elif content['name'] == 'stderr': stderr.append(content['text'])
                elif msg_type == 'error':
                    stderr.append(f"{content['ename']}: {content['evalue']}")
                elif msg_type == 'status' and content['execution_state'] == 'idle':
                    break
            except queue.Empty:
                stderr.append("Timeout")
                break
        
        return {"stdout": "".join(stdout), "stderr": "".join(stderr)}

    def restart(self):
        self.cleanup()
        self._start_kernel()

    def cleanup(self):
        if self.kc: 
            try: self.kc.stop_channels()
            except: pass
        if self.km: 
            try: self.km.shutdown_kernel(now=True)
            except: pass
        
        # 임시 디렉토리 청소
        if self.connection_dir and os.path.exists(self.connection_dir):
            shutil.rmtree(self.connection_dir, ignore_errors=True)
            
        self.kc = None
        self.km = None
        self.connection_dir = None


class AgentSandbox:
    def __init__(self, work_dir="./"):
        self.work_dir = work_dir
        
        # 1. Main Kernel: 데이터와 상태를 계속 유지 (Solver용)
        logger.info("🟢 Starting Main Kernel...")
        self.main_kernel = SingleKernel(work_dir)
        
        # 2. Tester Kernel: 언제든 버릴 수 있는 검증용 (Tester용)
        os.makedirs(TEST_DIR, exist_ok=True)
        logger.info("🟡 Starting Tester Kernel...")
        self.test_kernel = SingleKernel(TEST_DIR)


    def copy_files_to_tester(self, file_names: list):
        """
        Main 폴더의 파일들을 Tester 폴더로 복사합니다. (데이터 파일 등)
        """
        for fname in file_names:
            src = os.path.join(self.work_dir, fname)
            dst = os.path.join(TEST_DIR, fname)
            
            if os.path.exists(src):
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
                logger.info(f"📂 Copied {fname} to test env.")
            else:
                logger.warning(f"⚠️ File not found in Main: {fname}")


    def run_code(self, code: str, mode: str = "permanent") -> dict:
        """
        mode에 따라 실행할 커널을 선택합니다.
        - 'permanent': Main Kernel에서 실행 (상태 저장됨)
        - 'temporary': Test Kernel에서 실행 (격리됨)
        """
        
        if mode == "temporary":
            # 🔥 Tester 커널에서 실행
            # 옵션: 매 테스트마다 커널을 재시작해서 '완전 순수 상태'를 보장할 수도 있음
            self.test_kernel.restart() # 너무 느리면 생략 가능
            
            logger.info("🧪 Running in TESTER Kernel (Isolated)")
            return self.test_kernel.execute(code)
            
        else: # permanent
            # 🔥 Main 커널에서 실행
            logger.info("💾 Running in MAIN Kernel (Stateful)")
            return self.main_kernel.execute(code)

    def cleanup(self):
        self.main_kernel.cleanup()
        self.test_kernel.cleanup()
        shutil.rmtree(TEST_DIR, ignore_errors=True)

    def cleanup_main_kernel(self):
        self.main_kernel.cleanup()

    def cleanup_test_kernel(self):
        self.test_kernel.cleanup()
        shutil.rmtree(TEST_DIR, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        # 에러를 숨기지 않고 전파하려면 False 반환 (또는 생략)
        return False
    
    def get_final_context(self):
        """
        최종 답변 생성을 위해 커널의 변수들을 가져옵니다.
        - Scalar (int, float, str): 값을 그대로 가져옴 (답변에 써야 하니까)
        - DataFrame/Series: 요약 정보만 가져옴 (답변의 근거용)
        - Model/Others: 존재 여부만 확인
        """
        inspection_code = """
import json
import pandas as pd
import numpy as np
import types

def _get_final_context():
    context = {
        "answers": {},   # 최종 답안 후보 (int, float, str)
        "evidence": {}   # 증거 자료 (DataFrame, Plot 등)
    }
    
    for name, val in list(globals().items()):
        if name.startswith('_') or name in ['In', 'Out', 'get_ipython', 'exit', 'quit', 'open']: 
            continue
        if isinstance(val, (types.ModuleType, types.FunctionType)): 
            continue

        # 1. 🎯 Answers: 스칼라 값 (숫자, 짧은 문자열) -> 직접적인 정답일 확률 99%
        if isinstance(val, (int, float, np.number)):
            context['answers'][name] = val
        elif isinstance(val, str) and len(val) < 200: # 너무 긴 문자열은 제외
            context['answers'][name] = val
        elif isinstance(val, (list, tuple)) and len(val) < 10: # 짧은 리스트
            context['answers'][name] = val
            
        # 2. 📊 Evidence: 데이터프레임 -> 요약 정보만 (Head)
        elif hasattr(val, 'head') and hasattr(val, 'shape'):
            # 마크다운으로 변환하여 LLM이 읽기 좋게
            preview = val.head(3).to_markdown() if hasattr(val, 'to_markdown') else str(val.head(3))
            context['evidence'][name] = {
                'type': type(val).__name__,
                'shape': str(val.shape),
                'preview': preview
            }
            
        # 3. 📦 Others: 그 외 (모델 객체 등) -> 타입만 표시
        else:
            context['evidence'][name] = {
                'type': type(val).__name__
            }
            
    return json.dumps(context, default=str)

print(_get_final_context())
"""
        result = self.main_kernel.execute(inspection_code)
        
        if result['stderr']:
            logger.error(f"Error inspecting variables: {result['stderr']}")
            return {}
            
        try:
            return json.loads(result['stdout'])
        except Exception as e:
            logger.error(f"Failed to parse inspection result: {e}")
            logger.debug(f"Raw stdout: {result['stdout']}")
            return {}