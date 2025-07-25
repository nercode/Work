from constant.consts import *
from dto.base_model_dto import BaseModelDTO
import csv

from util.path_util import PathUtil


class TokenCounterUtil:

	@staticmethod
	def count_tokens(model_dto: BaseModelDTO, item: str):
		model = model_dto.name
		input_tokens = model_dto.input_tokens
		output_tokens = model_dto.output_tokens

		data = [item, model, input_tokens, output_tokens]

		# write to csv file
		with open(PathUtil.get_project_root() / DATASET_DIR / TOKENS_FILE, mode='a', newline='', encoding='utf-8') as file:
			writer = csv.writer(file)

			# 写入单行数据
			writer.writerow(data)

		print("token_counter finish")
