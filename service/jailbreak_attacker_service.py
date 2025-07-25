import json
from typing import Dict, List
import transformers
import torch

from constant.consts import (
	LLAMA_3_1_8B_MODEL, TYPE1_ATTACKER_PROMPT,
	TYPE2_ATTACKER_PROMPT,
)
from service.base_model_service import BaseModelService


class JailBreakAttackerService:
	"""
	execute jailbreak attack
	"""

	def __init__(self, base_model_service: BaseModelService):
		self.base_model_service = base_model_service


	@staticmethod
	def execute_attacks(query: dict):
		intent = query.get("intent")
		context = query.get("context")

		pipeline = transformers.pipeline(
			"text-generation",
			model=LLAMA_3_1_8B_MODEL,
			model_kwargs={"torch_dtype": torch.bfloat16},
			device_map="auto",
		)

		if '+' in intent:
			system_prompt = TYPE2_ATTACKER_PROMPT.format(sample=intent.split('+')[1])
		else:
			system_prompt = TYPE1_ATTACKER_PROMPT

		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": context}
		]

		outputs = pipeline(
			messages,
			max_new_tokens=256,
		)
		print(outputs[0]["generated_text"][-1])

		# 返回最终结果
		return outputs[0]["generated_text"][-1]