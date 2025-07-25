import os

from dotenv import load_dotenv

from constant.consts import *
from dto.deepseek_model_dto import DeepSeekModelDTO
from dto.glm_model_dto import GLMModelDTO
from dto.llama_model_dto import LlamaModelDTO
from dto.moonshot_model_dto import MoonShotModelDTO
from service.openai_service import OpenAIService
from util.file_util import FileUtil

# 文件处理
from util.path_util import PathUtil
from dto.qwen_model_dto import QwenModelDTO
from dto.gpt_model_dto import GPTModelDTO
from service.query_parser_service import QueryParserService

from util.token_counter_util import TokenCounterUtil


# load .env file
load_dotenv()

project_root = PathUtil.get_project_root()


# The performance difference among various LLMs is not significant
gpt_model_dto = GPTModelDTO(GPT_4O_MINI_MODEL, "", os.getenv(GPT_API_KEY), 8192, 0, 0)
qwen_model_dto = QwenModelDTO(QWEN3_235B_A22B_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
# llama_model_dto = LlamaModelDTO(LLAMA_4_MAVERICK_MODEL, MODELSCOPE_URL, os.getenv(MODELSCOPE_API_KEY), 8192, 0, 0)
llama_model_dto = LlamaModelDTO(LLAMA_3_1_405B_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
glm_model_dto = GLMModelDTO(GLM_4_PLUS_MODEL, GLM_URL, os.getenv(GLM_API_KEY), 8192, 0, 0)
deepseek_model_dto = DeepSeekModelDTO(DEEPSEEK_CHAT_MODEL, DEEPSEEK_URL, os.getenv(DEEPSEEK_API_KEY), 8192, 0, 0)
# deepseek_model_dto = DeepSeekModelDTO(DEEPSEEK_R1_MODEL, MODELSCOPE_URL, os.getenv(MODELSCOPE_API_KEY), 8192, 0, 0)
moonshot_model_dto = MoonShotModelDTO(MOONSHOT_V1_32K_MODEL, MOONSHOT_URL, os.getenv(MOONSHOT_API_KEY), 8192, 0, 0)

base_model_service = OpenAIService(qwen_model_dto)
# base_model_service = OpenAIService(llama_model_dto)
# base_model_service = OpenAIService(gpt_model_dto)
# base_model_service = OpenAIService(glm_model_dto)
# base_model_service = OpenAIService(deepseek_model_dto)
# base_model_service = OpenAIService(moonshot_model_dto)

query_parser_service = QueryParserService(base_model_service)

# prepare queries
queries_file = project_root / DATASET_DIR / QUERIES_FILE
queries = FileUtil.read_data_from_jsonl(queries_file)

def test_query_parser():
	updated_queries = query_parser_service.parse_queries(queries, 20)
	# 将更新后的问题写回文件
	FileUtil.write_data_to_jsonl(queries_file, updated_queries)
	print("问题处理完成并已写回文件")
	TokenCounterUtil.count_tokens(base_model_service.model_dto, "query_parser")


test_query_parser()