from dataclasses import dataclass

@dataclass
class BaseModelDTO:
	"""模型父类"""

	# 模型名称
	name: str
	# 模型url
	url: str
	# 模型API_KEY
	api_key: str
	# 模型最大输出长度
	max_output: int
	# 输入token
	input_tokens: int
	# 输出token
	output_tokens: int