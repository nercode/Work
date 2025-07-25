import json
from typing import Dict, List

from constant.consts import *
from service.base_model_service import BaseModelService


class QueryParserService:
	"""
	classify and extract the queries
	"""

	def __init__(self, base_model_service: BaseModelService):
		self.base_model_service = base_model_service


	def parse_queries(self, queries: List[Dict], batch_size: int = 20) -> List[Dict]:
		"""
		parse harmful queries

		:param queries: harmful queries
		:param batch_size: number of queries per batch
		"""
		# 更新后的问题
		updated_queries = []
		for i in range(0, len(queries), batch_size):
			# construct few_shot + queries
			batch = queries[i:i + batch_size]
			batch_queries = [{"index": query.get("index"), "instruction": query.get("query")} for query in batch]
			messages = [
				{"role": "system", "content": PARSER_PROMPT},
				{"role": "user", "content": PARSER_PREFIX + json.dumps(batch_queries, ensure_ascii=False)}
			]

			results = self.base_model_service.get_response(
				messages=messages,
				response_format={"type": "json_object"},
				extra_body={"enable_thinking": False},
			)
			if results:
				# 提高匹配效率
				query_dict = {query.get("index"): query for query in batch}
				for result in results:
					# 获取返回结果中的 index
					result_index = result.get('index')
					# LLM返回的问题与batch中的问题做匹配, 避免更新错
					if result_index in query_dict:
						query_dict[result_index]['type'] = result.get('type')
						query_dict[result_index]['intent'] = result.get('intent')
						print(f"index:{result_index},type:{result.get('type')}")
						print(f"intent:{result.get('intent')}")
			else:
				print(f"process No.{i // batch_size + 1} batch queries, LLM returns null, skip this batch")
			# 更新后的问题加入集合
			updated_queries.extend(batch)
		# 返回最终结果
		return updated_queries