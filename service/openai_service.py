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