from abc import ABC, abstractmethod
from typing import Any

from dto.base_model_dto import BaseModelDTO


class BaseModelService(ABC):

	def __init__(self, base_model_dto: BaseModelDTO):
		self.model_dto = base_model_dto
		self.client = None
		self.init_client()


	@abstractmethod
	def init_client(self):
		pass


	@abstractmethod
	def get_response(self, **kwargs: Any):
		pass