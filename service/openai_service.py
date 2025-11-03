import json
from typing import Any

from openai import OpenAI

from service.base_model_service import BaseModelService


class OpenAIService(BaseModelService):

	def init_client(self):
		# gpt LLMs
		if not self.model_dto.url:
			self.client = OpenAI(api_key=self.model_dto.api_key)
		# other LLMs
		else:
			self.client = OpenAI(
				api_key=self.model_dto.api_key,
				base_url=self.model_dto.url,
			)


	def get_response(self, **kwargs: Any):
		try:
			response = self.client.chat.completions.create(
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