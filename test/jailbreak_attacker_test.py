import os

from dotenv import load_dotenv

from constant.consts import *
from dto.deepseek_model_dto import DeepSeekModelDTO
from dto.glm_model_dto import GLMModelDTO
from dto.llama_model_dto import LlamaModelDTO
from service.openai_service import OpenAIService
from service.jailbreak_attacker_service import JailBreakAttackerService
from util.file_util import FileUtil

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

qwen_model_dto = QwenModelDTO(QWEN_PLUS_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
llama_model_dto = LlamaModelDTO(LLAMA_3_3_70B_MODEL, MODELSCOPE_URL, os.getenv(MODELSCOPE_API_KEY), 8192, 0, 0)
gpt_model_dto = GPTModelDTO(GPT_4O_MINI_MODEL, "", os.getenv(GPT_API_KEY), 8192, 0, 0)
glm_model_dto = GLMModelDTO(GLM_4_PLUS_MODEL, GLM_URL, os.getenv(GLM_API_KEY), 8192, 0, 0)
deepseek_model_dto = DeepSeekModelDTO(DEEPSEEK_CHAT_MODEL, DEEPSEEK_URL, os.getenv(DEEPSEEK_API_KEY), 8192, 0, 0)

base_model_service = OpenAIService(qwen_model_dto)
# base_model_service = OpenAIService(llama_model_dto)
# base_model_service = OpenAIService(gpt_model_dto)
# base_model_service = OpenAIService(glm_model_dto)
# base_model_service = OpenAIService(deepseek_model_dto)

jailbreak_attacker_service = JailBreakAttackerService(base_model_service)


def test_execute_attack():
	# 执行攻击
	for query in queries:
		response = jailbreak_attacker_service.execute_attacks(query)
		query["response"] = response
		print(f"问题:{query.get('index')},回复:{response}")

	print("所有问题测试完成")
	# 写回文件
	FileUtil.write_data_to_jsonl(queries_file, queries)
	print(f"写回文件完成")
	TokenCounterUtil.count_tokens(base_model_service.model_dto, "jailbreak_attacker")
	print("token计算完成")


test_execute_attack()