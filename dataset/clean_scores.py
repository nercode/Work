import json


def clear_scores_in_jsonl(file_path):
	# 临时存储修改后的行
	modified_lines = []

	# 读取并处理每一行
	with open(file_path, 'r', encoding='utf-8') as file:
		for line in file:
			# 解析 JSON 行
			obj = json.loads(line)

			# 检查是否存在 'scores' 键，并清空其值
			if 'scores' in obj:
				obj['scores'] = []

			# # 检查是否存在 'context_quality' 键，并清空其值
			# if 'context_quality' in obj:
			# 	obj['context_quality'] = -1

			# 将修改后的对象转换回 JSON 字符串并添加到列表中
			modified_lines.append(json.dumps(obj, ensure_ascii=False))

	# 写入修改后的内容到文件
	with open(file_path, 'w', encoding='utf-8') as file:
		for line in modified_lines:
			file.write(line + '\n')


# 示例调用
clear_scores_in_jsonl('queries.jsonl')