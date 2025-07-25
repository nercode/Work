import os

from dotenv import load_dotenv

from constant.consts import *
from dto.deepseek_model_dto import DeepSeekModelDTO
from dto.glm_model_dto import GLMModelDTO
from service.openai_service import OpenAIService
from service.prompt_evaluator_service import PromptEvaluatorService
from util.file_util import FileUtil

# 文件处理
from util.path_util import PathUtil
from dto.qwen_model_dto import QwenModelDTO
from dto.gpt_model_dto import GPTModelDTO
from util.token_counter_util import TokenCounterUtil


# 加载.env文件
load_dotenv()

project_root = PathUtil.get_project_root()

# prepare queries
queries_file = project_root / DATASET_DIR / QUERIES_FILE
queries = FileUtil.read_data_from_jsonl(queries_file)

# qwen-plus效果最好
qwen_model_dto = QwenModelDTO(QWEN_PLUS_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 129024, 0, 0)
gpt_model_dto = GPTModelDTO(GPT_4O_MINI_MODEL, "", os.getenv(GPT_API_KEY), 129024, 0, 0)
glm_model_dto = GLMModelDTO(GLM_4_PLUS_MODEL, GLM_URL, os.getenv(GLM_API_KEY), 129024, 0, 0)
deepseek_model_dto = DeepSeekModelDTO(DEEPSEEK_CHAT_MODEL, DEEPSEEK_URL, os.getenv(DEEPSEEK_API_KEY), 129024, 0, 0)

base_model_service = OpenAIService(qwen_model_dto)
# base_model_service = OpenAIService(gpt_model_dto)
# base_model_service = OpenAIService(glm_model_dto)
# base_model_service = OpenAIService(deepseek_model_dto)

context_evaluator_service = PromptEvaluatorService(base_model_service)


def test_evaluate_contexts():
	updated_queries = context_evaluator_service.evaluate_prompts(queries, 10, 5)

	print("所有问题质量评估完成")
	# 写回文件
	FileUtil.write_data_to_jsonl(queries_file, updated_queries)
	print("写回文件完成")
	TokenCounterUtil.count_tokens(base_model_service.model_dto, "quality_evaluator")
	print("token计算完成")


def test_compute_scores():
	# 计算每个问题的context评分
	for query in queries:
		query["context_quality"] = PromptEvaluatorService.compute_scores(query)

	print("context评分计算完成")
	# 写回文件
	FileUtil.write_data_to_jsonl(queries_file, queries)
	print(f"写回文件完成")

# test_compare_query_and_simplified()
# test_evaluate_contexts()
test_compute_scores()