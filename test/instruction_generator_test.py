import json
from typing import Dict, List

from constant.consts import GENERATOR_PREFIX

from service.base_model_service import BaseModelService


class InstructionGeneratorService:
	"""
	generate context
	"""

	def __init__(self, base_model_service: BaseModelService):
		self.base_model_service = base_model_service


	@staticmethod
	def prepare_few_shot(data: List[Dict]) -> List[Dict]:
		return data


	def generate_contexts(self, few_shot: List[Dict], queries: List[Dict], batch_size: int = 10) -> List[Dict]:
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
				"topic": f"{self.prefix_mapping.get(query.get('type'))}{query.get('intent')}"
			} for query in batch]
			combined_batch_queries = GENERATOR_PREFIX + json.dumps(batch_queries, ensure_ascii=False)
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
			else:
				print(f"process No.{i // batch_size + 1} batch queries, LLM returns null, skip this batch")
			# 更新后的问题加入集合
			updated_queries.extend(batch)
		# 返回最终结果
		return updated_queries