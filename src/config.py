import os
from dotenv import load_dotenv

# 경로 설정
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(BASE_DIR, "data", "logs")
DB_DIR = os.path.join(BASE_DIR, "data", "tool_db")
DATASET_DIR = os.path.join(BASE_DIR, "data", "dataset")
RESULT_DIR = os.path.join(BASE_DIR, "data", "result")
TEST_DIR = os.path.join(BASE_DIR, "test_env")

# 디렉토리 자동 생성
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# vLLM 서버 설정
# VLLM_BASE_URL = "http://localhost:8000/v1"
# VLLM_API_KEY = "EMPTY"  # 로컬 vLLM은 보통 키가 필요 없음
# MODEL_NAME = "Qwen/Qwen3-8B"  # 띄워놓은 모델명과 일치해야 함

load_dotenv(os.path.join(BASE_DIR, ".env"))

BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.environ.get("OPENROUTER_API_KEY")
# MODEL_NAME = "meta-llama/llama-3.3-70b-instruct"
MODEL_NAME = "google/gemini-2.5-flash"