import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List

from dto.base_model_dto import BaseModelDTO


class BaseModelService(ABC):

	def __init__(self, base_model_dto: BaseModelDTO):
		self.model_dto = base_model_dto
		self.client = None
		self.init_client()


	@abstractmethod
	def init_client(self):
		pass


	def get_response(self, **kwargs: Any):
		try:
			response = self.client.chat.completions.create(
				max_tokens=self.model_dto.max_output,
				model = self.model_dto.name,
				**kwargs
			)

			# 计算消耗的token
			self.model_dto.input_tokens += response.usage.prompt_tokens
			self.model_dto.output_tokens += response.usage.completion_tokens

			if response.choices and response.choices[0].message.content:
				# return json or [jsons]
				if 'response_format' in kwargs and kwargs.get('response_format').get('type') == 'json_object':
					try:
						return json.loads(response.choices[0].message.content)
					except json.JSONDecodeError as e:
						print(f"JSON decoding error:{e}, LLM returns:{response.choices[0].message.content}")
						return None

				# return str
				return response.choices[0].message.content
			else:
				print("LLM returns null")
				return None
		except Exception as e:
			print(f"call to LLM failed: {str(e)}")
			return None