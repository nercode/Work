import json
from typing import Any

from google import genai

from service.base_model_service import BaseModelService


class GenAIService(BaseModelService):

	def init_client(self):
		# gemini LLMs
		if not self.model_dto.url:
			self.client = genai.Client(api_key=self.model_dto.api_key)


	def get_response(self, **kwargs: Any):
		try:
			response = self.client.models.generate_content(
				model=self.model_dto.name,
				**kwargs
			)

			return response.text
		except Exception as e:
			print(f"call to LLM failed: {str(e)}")
			return None