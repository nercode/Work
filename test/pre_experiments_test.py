import os
import random
import time

from dotenv import load_dotenv

from constant.consts import *
from dto.deepseek_model_dto import DeepSeekModelDTO
from dto.glm_model_dto import GLMModelDTO
from dto.llama_model_dto import LlamaModelDTO
from dto.moonshot_model_dto import MoonShotModelDTO
from service.openai_service import OpenAIService
from util.file_util import FileUtil

from typing import List, Dict, Union

# 文件处理
from util.path_util import PathUtil
from dto.qwen_model_dto import QwenModelDTO
from dto.gpt_model_dto import GPTModelDTO
from service.pre_experiments_service import PreExperimentsService

from util.token_counter_util import TokenCounterUtil
from util.visualization_util import VisualizationUtil


# 加载.env文件
load_dotenv()

project_root = PathUtil.get_project_root()


# The performance difference among various LLMs is not significant
gpt_model_dto = GPTModelDTO(GPT_4O_MINI_MODEL, "", os.getenv(GPT_API_KEY), 8192, 0, 0)
qwen_model_dto = QwenModelDTO(QWEN_TURBO_20250715_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
# qwen_model_dto = QwenModelDTO(QWEN_PLUS_20250428, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
# qwen_model_dto = QwenModelDTO(QWEN3_235B_A22B_INSTRUCT_MODEL, QWEN_URL, os.getenv(QWEN_API_KEY), 8192, 0, 0)
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

pre_experiments_service = PreExperimentsService(base_model_service)

# prepare queries
queries_file = project_root / DATASET_DIR / PRE_EXPERIMENTS_DIR / QUERIES_FILE
queries = FileUtil.read_data_from_jsonl(queries_file)

# results_file = project_root / DATASET_DIR / PRE_EXPERIMENTS_DIR / RESULTS_FILE
results_file = project_root / DATASET_DIR / RESULTS_FILE

def test_generate_scenarios():
	# randomly select 20
	random_queries = random.sample(queries, 20)
	results = []
	for query in random_queries:
		index = query.get("index")
		query_content = query.get("query")
		print(f"index:{index}")

		scenarios = pre_experiments_service.generate_scenarios(query_content)
		results.append({"index": index, "query": query_content, "feature": "B", "scenario": ""})
		results.append({"index": index, "query": query_content, "feature": "N", "scenario": scenarios.get("N")})
		results.append({"index": index, "query": query_content, "feature": "R", "scenario": scenarios.get("R")})
		results.append({"index": index, "query": query_content, "feature": "RT", "scenario": scenarios.get("RT")})

		# 结果写回文件
		FileUtil.write_data_to_csv_append(results_file, results)
		results.clear()


def test_execute_attacks():
	results = FileUtil.read_data_from_csv(results_file)
	for result in results:
		result["response"] = pre_experiments_service.execute_attacks(result.get("query"), result.get("scenario"))
		print(f"{result['index']}: {result['response']}")
	FileUtil.write_data_to_csv_cover(results_file, results)


def test_evaluate_asr_w():
	results = FileUtil.read_data_from_csv(results_file)
	for result in results:
		result["asr_w"] = PreExperimentsService.evaluate_asr_w(result.get("response"), ASR_W)
	FileUtil.write_data_to_csv_cover(results_file, results)


def test_evaluate_asr_g():
	results = FileUtil.read_data_from_csv(results_file)
	for result in results:
		judgment = pre_experiments_service.evaluate_asr_g(result.get("query"), result.get("response"))
		result["asr_g_reason"], result["asr_g_score"] = PreExperimentsService.extract_reason_and_score(judgment)
		print(f"{result['index']}: {result['asr_g_score']}")

	FileUtil.write_data_to_csv_cover(results_file, results)


def test_evaluate_features(text_type: str):
	results = FileUtil.read_data_from_csv(results_file)
	for result in results:
		if text_type == "scenario":
			if result.get("feature") != "B":
				time.sleep(1)
				result["feature_score_scenario"] = pre_experiments_service.evaluate_features(result.get("query"), result.get("scenario"))
				print(f"{result['index']}: {result['feature']}: {result['feature_score_scenario']}")

		if text_type == "prompt":
			time.sleep(1)
			query_content = result.get("query")
			text = TABLE_ATTACKER_PROMPT.format(query=query_content, scenario=result.get("scenario"))
			result["feature_score_prompt"] = pre_experiments_service.evaluate_features(query_content, text)
			print(f"{result['index']}: {result['feature']}: {result['feature_score_prompt']}")

	FileUtil.write_data_to_csv_cover(results_file, results)


def test_calculate_feature_score(feature_score: str):
	results = FileUtil.read_data_from_csv(results_file)
	avg_feature_scores = PreExperimentsService.calculate_feature_score(results, feature_score)
	for feature, avg_scores in avg_feature_scores.items():
		print(f"Feature: {feature}, avg_score_relevance: {avg_scores[0]:.2f}, avg_score2_toxicity: {avg_scores[1]:.2f}")


def test_plot_pre_asr_w():
	VisualizationUtil.plot_pre_asr_w(results_file)


def test_plot_pre_asr_g_score():
	VisualizationUtil.plot_pre_asr_g_score(results_file)


# test_generate_scenarios()

# test_execute_attacks()

# test_evaluate_asr_w()
# test_evaluate_asr_g()

# test_evaluate_features("scenario")
# test_evaluate_features("prompt")

# test_calculate_feature_score("feature_score_prompt")
# test_calculate_feature_score("feature_score_scenario")

# test_plot_pre_asr_w()
# test_plot_pre_asr_g_score()


def test_evaluate(text_type: str):
	results = FileUtil.read_data_from_csv(results_file)
	for result in results:
		if text_type == "scenario":
			time.sleep(1)
			result["feature_score_scenario"] = pre_experiments_service.evaluate_features(result.get("query"), result.get("scenario"))
			print(f"{result['index']}: {result['feature_score_scenario']}")

		if text_type == "prompt":
			time.sleep(1)
			query_content = result.get("query")
			text = TABLE_ATTACKER_PROMPT.format(query=query_content, scenario=result.get("scenario"))
			result["feature_score_prompt"] = pre_experiments_service.evaluate_features(query_content, text)
			print(f"{result['index']}: {result['feature_score_prompt']}")

	FileUtil.write_data_to_csv_cover(results_file, results)

# test_evaluate("scenario")
test_calculate_feature_score("feature_score_scenario")