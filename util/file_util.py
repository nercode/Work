import json
import csv
from typing import Dict, List
import pandas as pd

from pathlib import Path


class FileUtil:

	@staticmethod
	def read_data_from_jsonl(file_path: Path) -> List[Dict]:
		"""
		read data from a jsonl file

		:param file_path: file path
		:return: data
		"""
		data = []
		try:
			with open(file_path, "r", encoding="utf-8") as f:
				for line in f:
					try:
						data.append(json.loads(line.strip()))
					except json.JSONDecodeError as e:
						print(f"JSON decode error in line: {line.strip()}, Error: {e}")
		except FileNotFoundError:
			print(f"File not found: {file_path}")
			return []
		return data


	@staticmethod
	def write_data_to_jsonl(file_path: Path, data: List[Dict]) -> None:
		"""
		write data to a jsonl file

        :param file_path: file path
        :param data: data
		"""
		with open(file_path, 'w', encoding='utf-8') as file:
			for query in data:
				file.write(json.dumps(query, ensure_ascii=False) + "\n")


	@staticmethod
	def read_data_from_csv(file_path: Path) -> List[Dict]:
		"""
		read data from a csv file

		:param file_path: file path
		:return: data
		"""
		if not file_path.is_file():
			print(f"错误：文件不存在于路径 '{file_path}'")
			return []

		try:
			# 使用 pandas 读取 CSV 文件。
			# to_dict('records') 会将 DataFrame 转换为字典列表，
			# 其中每个字典代表一行，键是列名。
			df = pd.read_csv(file_path)
			return df.to_dict('records')
		except pd.errors.EmptyDataError:
			print(f"警告：文件 '{file_path}' 是空的。")
			return []
		except pd.errors.ParserError as e:
			print(f"解析 CSV 文件 '{file_path}' 时出错：{e}")
			return []
		except Exception as e:
			print(f"读取文件 '{file_path}' 时发生未知错误：{e}")
			return []


	@staticmethod
	def write_data_to_csv_append(file_path: Path, data: List[Dict]) -> None:
		"""
		write data to a CSV file with append mode.

		:param file_path: Path to the CSV file.
		:param data: A list of dictionaries where each dictionary represents a row in the CSV.
		"""
		if not data:
			raise ValueError("Data is empty. Cannot write empty data to CSV.")

		with open(file_path, 'a', newline='', encoding='utf-8') as csvfile:
			# Use the keys from the first dictionary as the header
			fieldnames = data[0].keys()
			writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

			# Write each row
			for row in data:
				writer.writerow(row)


	@staticmethod
	def write_data_to_csv_cover(file_path: Path, data: List[Dict]):
		"""
		write data to a CSV file with cover mode.

		:param file_path: The path to the CSV file where data will be written.
		:param data: A list of dictionaries, where each dictionary represents a row.
		"""
		if not data:
			print(f"警告：没有数据可以写入文件 '{file_path}'。")
			# 可以选择创建空文件或不执行任何操作
			pd.DataFrame().to_csv(file_path, index=False)
			return

		try:
			# 将字典列表转换为 pandas DataFrame
			df = pd.DataFrame(data)

			# 将 DataFrame 写入 CSV 文件。
			# index=False 阻止 pandas 将 DataFrame 索引写入 CSV 文件作为第一列。
			df.to_csv(file_path, index=False)
			print(f"数据已成功写入 '{file_path}'。")
		except Exception as e:
			print(f"写入文件 '{file_path}' 时发生错误：{e}")
