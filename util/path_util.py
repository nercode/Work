from pathlib import Path


class PathUtil:

	@staticmethod
	def get_project_root() -> Path:
		current_file_path = Path(__file__).resolve()
		project_root = current_file_path.parents[1]
		return project_root
