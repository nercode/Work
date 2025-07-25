import json
from pathlib import Path
from typing import Dict, List

import torch

from constant.consts import GENERATOR_PREFIX_OFFLINE, GENERATOR_PREFIX_ONLINE

from service.base_model_service import BaseModelService

from transformers import AutoModelForCausalLM, AutoTokenizer


class ContextGeneratorService:
	"""
	generate context
	"""

	def __init__(self, base_model_service: BaseModelService):
		self.base_model_service = base_model_service


	@staticmethod
	def prepare_few_shot(data: List[Dict]) -> List[Dict]:
		return data


	def generate_contexts_with_online_model(self, few_shot: List[Dict], queries: List[Dict], batch_size: int = 10) -> List[Dict]:
		"""
		process harmful queries

		:param few_shot: few_shot
		:param queries: harmful queries
		:param batch_size: number of queries generated per batch
		"""
		# 更新后的问题
		updated_queries = []
		for i in range(0, len(queries), batch_size):
			# construct few_shot + queries
			batch = queries[i:i + batch_size]
			batch_queries = [{
				"index": query.get("index"),
				"topic": "steps of " + query.get('intent').split('+')[0] if '+' not in query.get('intent')
				else "examples of " + query.get('intent').split('+')[0]
			} for query in batch]
			combined_batch_queries = GENERATOR_PREFIX_ONLINE + json.dumps(batch_queries, ensure_ascii=False)
			messages = [
				*few_shot,
				{"role": "user", "content": combined_batch_queries}
			]

			results = self.base_model_service.get_response(
				messages=messages,
				response_format={"type": "json_object"},
			)
			if results:
				# 提高匹配效率
				query_dict = {query.get("index"): query for query in batch}

				for result in results:
					# 获取返回结果中的 index
					result_index = result.get('index')
					# LLM返回的问题与batch中的问题做匹配, 避免更新错
					if result_index in query_dict:
						query_dict[result_index]['context'] = result.get('report', "")
						print(f"index:{result_index}")
						print(f"query:{query_dict.get(result_index)}")
						print(f"context:{result.get('report', '')}")
			else:
				print(f"process No.{i // batch_size + 1} batch queries, LLM returns null, skip this batch")
			# 更新后的问题加入集合
			updated_queries.extend(batch)
		# 返回最终结果
		return updated_queries


	@staticmethod
	def generate_contexts_with_offline_model(model_path: Path, few_shot: List[Dict], queries: List[Dict], batch_size: int = 10) -> List[Dict]:
		"""
		process harmful queries

		:param model_path: model
		:param few_shot: few_shot
		:param queries: harmful queries
		:param batch_size: number of queries generated per batch
		"""
		# 更新后的问题
		updated_queries = []
		for i in range(0, len(queries)):
			messages = [
				*few_shot,
				{"role": "user", "content": f"{GENERATOR_PREFIX_OFFLINE}.Topic:{queries[i].get('intent').split('+')[0]}"}
			]

			# load the tokenizer and the model
			tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
			model = AutoModelForCausalLM.from_pretrained(
				str(model_path),
				torch_dtype="auto",
				device_map="auto",
				trust_remote_code=True
			).to("cuda" if torch.cuda.is_available() else "cpu")

			text = tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
				# Switches between thinking and non-thinking modes. Default is True.
				enable_thinking=False
			)
			model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

			# conduct text completion
			generated_ids = model.generate(
				**model_inputs,
				max_new_tokens=32768
			)

			# 去掉输入部分，只保留生成的内容
			output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

			# 非思考模式：直接解码输出
			content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

			print("content:", content)

			queries[i]["intent"] = content

			# 更新后的问题加入集合
			updated_queries.extend(queries[i])
		# 返回最终结果
		return updated_queries