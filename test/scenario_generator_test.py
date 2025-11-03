import os

from dotenv import load_dotenv

from constant.consts import *
from dto.glm_model_dto import GLMModelDTO
from dto.gpt_model_dto import GPTModelDTO
from dto.qwen_model_dto import QwenModelDTO
from dto.deepseek_model_dto import DeepSeekModelDTO
from service.openai_service import OpenAIService
from service.scenario_generator_service import ScenarioGeneratorService
from util.file_util import FileUtil
from util.path_util import PathUtil
from util.token_counter_util import TokenCounterUtil


# 加载.env文件
load_dotenv()

project_root = PathUtil.get_project_root()

# QWEN_TURBO_MODEL QWEN_PLUS_MODEL QWEN_MAX_MODEL QWEN_V2_5_72B_INSTRUCT_MODEL
qwen_model_dto = QwenModelDTO(QWEN_PLUS_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 16384, 0, 0)
deepseek_model_dto = DeepSeekModelDTO(DEEPSEEK_CHAT_MODEL, DEEPSEEK_URL, os.getenv(DEEPSEEK_API_KEY), 16384, 0, 0)
glm_model_dto = GLMModelDTO(GLM_4_PLUS_MODEL, GLM_URL, os.getenv(GLM_API_KEY), 16384, 0, 0)
gpt_model_dto = GPTModelDTO(GPT_4O_MINI_MODEL, "", os.getenv(GPT_API_KEY), 16384, 0, 0)

# GLM效果最好
# base_model_service = OpenAIService(qwen_model_dto)
# base_model_service = OpenAIService(deepseek_model_dto)
base_model_service = OpenAIService(glm_model_dto)
# base_model_service = OpenAIService(gpt_model_dto)

scenario_generator_service = ScenarioGeneratorService(base_model_service)

# prepare few_shot
few_shot_file = project_root / DATASET_DIR / SCENARIO_GENERATOR_FILE
data = FileUtil.read_data_from_jsonl(few_shot_file)
few_shot = scenario_generator_service.prepare_few_shot(data)

# prepare queries
queries_file = project_root / DATASET_DIR / QUERIES_FILE
queries = FileUtil.read_data_from_jsonl(queries_file)

# prepare model
model = project_root / MODEL_DIR / QWEN3_0_6B_DIR

def test_process_queries():
	updated_queries = scenario_generator_service.generate_scenarios_with_online_model(few_shot, queries, 10)
	# updated_queries = scenario_generator_service.generate_contexts_with_offline_model(model, few_shot, queries, 10)
	# 显示结果
	print(updated_queries)
	# 将更新后的问题写回文件
	FileUtil.write_data_to_jsonl(queries_file, updated_queries)
	print("问题处理完成并已写回文件")
	TokenCounterUtil.count_tokens(base_model_service.model_dto, "context_generator")
	print("token计算完成")

test_process_queries()