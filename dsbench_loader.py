import json
import os
import glob
import pandas as pd



def find_jpg_files(directory):
    jpg_files = [file for file in os.listdir(directory) if file.lower().endswith('.jpg') or file.lower().endswith('.png')]
    return jpg_files if jpg_files else None

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def find_excel_files(directory):
    excel_files = [file for file in os.listdir(directory) if (file.lower().endswith('xlsx') or file.lower().endswith('xlsb') or file.lower().endswith('xlsm')) and not "answer" in file.lower()]
    return excel_files if excel_files else None

def read_excel(file_path):
    xls = pd.ExcelFile(file_path)
    sheets = {}
    for sheet_name in xls.sheet_names:
        sheets[sheet_name] = xls.parse(sheet_name)
    return sheets

def dataframe_to_text(df):
    text = df.to_string(index=False)
    return text

def combine_sheets_text(sheets):
    combined_text = ""
    for sheet_name, df in sheets.items():
        sheet_text = dataframe_to_text(df)
        combined_text += f"Sheet name: {sheet_name}\n{sheet_text}\n\n"
    return combined_text

def read_txt(path):
    with open(path, "r") as f:
        return f.read()

def truncate_text(text, max_tokens=128000):
    tokens = text.split()
    if len(tokens) > max_tokens:
        text = ' '.join(tokens[-max_tokens:])
    return text


class DSBenchLoader:
    def __init__(self, root_path, mode="analysis"):
        self.root_path = os.path.abspath(root_path)
        self.mode = mode
        self.dataset = [] 
        
        # --- 1. 경로 설정 ---
        if mode == "analysis":
            self.base_dir = os.path.join(self.root_path, "data_analysis")
            self.json_path = os.path.join(self.base_dir, "data.json")
            self.data_source_dir = os.path.join(self.base_dir, "data")
            
        elif mode == "modeling":
            self.base_dir = os.path.join(self.root_path, "data_modeling")
            self.json_path = os.path.join(self.base_dir, "data.json")
            self.data_resplit_dir = os.path.join(self.base_dir, "data", "data_resplit")
            self.task_desc_dir = os.path.join(self.base_dir, "data", "task")
            self.answers_dir = os.path.join(self.base_dir, "data", "answers")
            
        else:
            raise ValueError("Mode must be 'analysis' or 'modeling'")

        # --- 2. 데이터 로드 (JSONL 호환성 강화) ---
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"Metadata not found: {self.json_path}")
            
        with open(self.json_path, 'r', encoding='utf-8') as f:
            try:
                # 1차 시도: 일반 JSON List로 읽기
                raw_data = json.load(f)
                # 만약 JSON 객체 하나만 있다면 리스트로 감싸기
                if not isinstance(raw_data, list):
                    raw_data = [raw_data]
            except json.JSONDecodeError:
                # 실패 시: JSONL(Line-delimited)로 다시 읽기
                f.seek(0)
                raw_data = []
                for line in f:
                    if line.strip(): # 빈 줄 제외
                        raw_data.append(json.loads(line))

        # --- 3. 데이터셋 구성 (Flattening) ---
        if mode == "analysis":
            for item in raw_data:
                question_ids = item.get('questions', [])
                for q_id in question_ids:
                    self.dataset.append({
                        "task_item": item,
                        "question_id": q_id,
                        "type": "analysis"
                    })
        elif mode == "modeling":
            for item in raw_data:
                self.dataset.append({
                    "task_item": item,
                    "question_id": None,
                    "type": "modeling"
                })
        
        print(f"[{mode.upper()}] Loader initialized. Total questions: {len(self.dataset)}")

    def get_problem(self, index):
        entry = self.dataset[index]
        item = entry["task_item"]
        q_id = entry["question_id"]
        
        if self.mode == "analysis":
            return self._parse_analysis_problem(item, q_id)
        else:
            return self._parse_modeling_problem(item)

    def _read_file_content(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read().strip()
        return "(Content not found)"

    def _parse_analysis_problem(self, item, target_q_id):
        p_id = item.get('id', '')
        try:
            folder_name = f"{int(p_id):08d}"
        except:
            folder_name = str(p_id)

        target_dir = os.path.join(self.data_source_dir, folder_name)
        
        # Introduction & Excel Info
        image = find_jpg_files(target_dir)
        if image:
            image = os.path.join(target_dir, image[0])

        excel_content = ""
        excels = find_excel_files(target_dir)
        excel_file_path_list = []
        if excels:
            for excel in excels:
                excel_file_path = os.path.join(target_dir, excel)
                excel_file_path_list.append(excel_file_path)
                sheets = read_excel(excel_file_path)
                combined_text = combine_sheets_text(sheets)
                excel_content += f"The excel file {excel} is: " + combined_text

        intro_text = self._read_file_content(os.path.join(target_dir, "introduction.txt"))
        # excel_files = glob.glob(os.path.join(target_dir, "*.xls*")) + glob.glob(os.path.join(target_dir, "*.csv"))
        # if excel_files:
        #     # 절대 경로로 변환하여 에이전트가 확실히 찾을 수 있게 함
        #     file_list_str = "\n".join([f"- {os.path.abspath(f)}" for f in excel_files])
        #     excel_section = (
        #         f"Files located at:\n{file_list_str}\n\n"
        #         f"(Note: The data content is NOT provided here directly due to size limits. "
        #         f"You MUST use Python (pandas/openpyxl) to load and inspect these files from the paths above.)"
        #     )
        # else:
        #     excel_section = "(No excel files found in this directory. Please inspect the directory using os.listdir)"

        # Question Text (개별 질문 파일 읽기)
        q_text = self._read_file_content(os.path.join(target_dir, f"{target_q_id}.txt"))
        
        # Ground Truth 찾기
        all_q_ids = item.get('questions', [])
        all_answers = item.get('answers', [])
        ground_truth = None
        if target_q_id in all_q_ids:
            idx = all_q_ids.index(target_q_id)
            if idx < len(all_answers):
                ground_truth = all_answers[idx]

        prompt = (
            f"You are a data analyst. I will give you a background introduction and a data analysis question. You must answer the question.\n"
            f"The introduction is detailed as follows.\n"
            f"<introduction>\n{intro_text}\n</introduction>\n\n"
            # f"The workbook is detailed as follows.\n"
            # f"<excel>\n{excel_content}\n</excel>\n\n"
            f"The question is detailed as follows.\n"
            f"<question>\n{q_text}\n</question>\n\n"
            f"Please answer the above question using Python code."
            f"Here is a list of excel files you can read:\n"
            f"{excel_file_path_list}"
        )

        if image:
            prompt += f"\n\nHere is an image file path:\n{image}"
        
        return {
            "id": f"{p_id}_{target_q_id}",
            "prompt": prompt,
            "target_dir": os.path.abspath(target_dir),
            "question_id": target_q_id,
            "ground_truth": ground_truth,
            "original_data": item
        }

    def _parse_modeling_problem(self, item):
        task_name = item.get('name')
        target_dir = os.path.join(self.data_resplit_dir, task_name)
        desc_path = os.path.join(self.task_desc_dir, f"{task_name}.txt")
        intro_text = self._read_file_content(desc_path)
        answer_path = os.path.join(self.answers_dir, task_name, "test_answer.csv")
        
        prompt = (
            f"You are a data scientist. I have a data modeling task. You must give me "
            f"the predicted results as a CSV file as detailed in the following content. "
            f"Please don’t ask me any questions. I provide you with three files. "
            f"One is training data, one is test data. There is also a sample file for submission.\n\n"
            f"The task introduction is detailed as follows.\n"
            f"<introduction>\n{intro_text}\n</introduction>\n\n"
            f"The training data, testing data, and sample of submission files are in path:\n"
            f"{os.path.abspath(target_dir)}"
        )

        return {
            "id": task_name,
            "prompt": prompt,
            "target_dir": os.path.abspath(target_dir),
            "ground_truth_path": answer_path if os.path.exists(answer_path) else None,
            "original_data": item
        }

    def __len__(self):
        return len(self.dataset)

# --------------------------------------------------------------------------------
# TEST CODE
# --------------------------------------------------------------------------------
if __name__ == "__main__":
    ROOT_DIR = "/c1/geonju/project/data/dataset/DSBench" 
    print(f"🚀 Testing DSBenchLoader with root: {os.path.abspath(ROOT_DIR)}")

    # TEST 1: Analysis Mode
    print("\n" + "="*60)
    print("📊 [TEST 1] Data Analysis Mode")
    print("="*60)
    try:
        loader = DSBenchLoader(ROOT_DIR, mode="analysis")
        if len(loader) > 1:
            p1 = loader.get_problem(0)
            p2 = loader.get_problem(1)
            print(f"✅ Loaded {len(loader)} questions.")
            print(f"🔹 Item 0 ID: {p1['id']}")
            print(f"🔹 Item 1 ID: {p2['id']}")

            print("=" * 50)
            print("Example Prompt")
            print(p1['prompt'])
            print("=" * 50)
            
            # Flattening 확인
            if p1['prompt'] != p2['prompt']:
                print("✅ Flattening SUCCESS: Prompts are different.")
            else:
                print("❌ Flattening FAILED: Prompts are same.")
    except Exception as e:
        print(f"❌ Analysis Failed: {e}")

    # TEST 2: Modeling Mode
    print("\n" + "="*60)
    print("🤖 [TEST 2] Data Modeling Mode")
    print("="*60)
    try:
        loader_mod = DSBenchLoader(ROOT_DIR, mode="modeling")
        if len(loader_mod) > 0:
            task = loader_mod.get_problem(0)
            print(f"✅ Loaded {len(loader_mod)} tasks.")
            print(f"🔹 Task: {task['id']}")
            print("=" * 50)
            print("Example Prompt")
            print(task['prompt'])
            print("=" * 50)
            if "Please don’t ask me" in task['prompt']:
                print("✅ Constraint Check: SUCCESS")
    except Exception as e:
        print(f"❌ Modeling Failed: {e}")