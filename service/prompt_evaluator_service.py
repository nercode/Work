import json
from typing import Dict, List

from sentence_transformers import SentenceTransformer
import torch

from pathlib import Path

from constant.consts import EVALUATOR_PROMPT
from service.base_model_service import BaseModelService


class PromptEvaluatorService:
	"""
	Evaluate the quality of prompts
	"""

	def __init__(self, base_model_service: BaseModelService):
		self.base_model_service = base_model_service


	def evaluate_prompts(self, queries: List[Dict], batch_size: int = 10, num: int = 5) -> List[Dict]:
		"""
		metrics:
			1.Relevance
			2.Coherence
			3.Toxicity
		tool: online model

		:param queries: harmful queries
		:param batch_size: number of contexts evaluated per batch
		:param num: repeats per context
		"""
		# 更新后的问题
		updated_queries = []
		for i in range(0, len(queries), batch_size):
			batch = queries[i:i + batch_size]

			# not consider query type
			batch_queries = [{
				"index": query.get("index"),
				"topic": query.get('intent'),
				"report": query.get("context"),
			} for query in batch]

			messages = [
				{"role": "system", "content": EVALUATOR_PROMPT},
				{"role": "user", "content": json.dumps(batch_queries, ensure_ascii=False)}
			]

			# 提高匹配效率
			query_dict = {query.get("index"): query for query in batch}

			# 每个问题多次评估
			for j in range(num):
				results = self.base_model_service.get_response(
					messages=messages,
					response_format={"type": "json_object"},
				)

				if results:
					for result in results:
						# 获取返回结果中的 index
						result_index = result.get('index')
						# LLM返回的问题与batch中的问题做匹配, 避免更新错
						if result_index in query_dict:
							query_dict[result_index]['scores'].append(json.loads(result.get('scores')))
				else:
					print(f"process No.{i // batch_size + 1} batch queries, LLM returns null, skip this batch")
			# 更新后的问题加入集合
			updated_queries.extend(batch)
		# 返回最终结果
		return updated_queries


	@staticmethod
	def compute_scores(data: dict) -> float:
		"""
		compute quality scores

		:param data: a query
		"""
		scores = data.get("scores")

		result = 0
		# 每个context有多次评分
		for score in scores:
			if min(score) < 0.7:
				return -1
			result += 0.5 * score[0] + 0.3 * score[1] + 0.2 * score[2]

		return result / len(scores) if result / len(scores) >= 0.7 else -1