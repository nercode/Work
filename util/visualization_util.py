import matplotlib
import pandas as pd
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from pathlib import Path


class VisualizationUtil:

	@staticmethod
	def plot_pre_asr_w(file_path: Path):
		"""
		factor_change
		:param file_path:
		:return:
		"""
		df = pd.read_csv(file_path, dtype={'asr_w': int})

		# Set font styles for the plot to 'Times New Roman' serif font.
		plt.rcParams['font.family'] = 'serif'
		plt.rcParams['font.serif'] = ['Times New Roman']
		# Allow negative signs to be displayed correctly, which is important for some plots.
		plt.rcParams['axes.unicode_minus'] = False

		# Step 1: Calculate the numerator - count of ASR_W=1 for each 'type'
		asr_one_counts = df[df['asr_w'] == 1]['feature'].value_counts().reindex(['N', 'R', 'RT'], fill_value=0)

		# Step 2: Calculate the denominator - total number of unique 'index' values
		# Count the number of unique values in the 'index' column.
		total_unique_indices = df['index'].nunique()

		# Handle the case where total_unique_indices might be zero to avoid division by zero.
		if total_unique_indices == 0:
			print("Warning: No unique 'index' values found in the data. Cannot calculate ASR_W.")
			return

		# Step 3: Calculate the Attack Success Rate (ASR_W) as a proportion and convert to percentage
		asr_percentages = (asr_one_counts / total_unique_indices) * 100

		# Step 4: Create the bar chart
		plt.figure(figsize=(4, 4))

		# Define the width of the bars for visual spacing.
		bar_width = 0.5

		bar_color = '#49A0AD'  # Hex for RGB(73, 160, 173)

		# Plot the bars using the calculated ASR_W percentages.
		bars = plt.bar(
			asr_percentages.index,
			asr_percentages.values,
			color=bar_color,
			width=bar_width
		)

		# Add percentage labels on top of each bar.
		for bar in bars:
			height = bar.get_height()  # Get the height (percentage value) of the current bar.
			if height > 0:  # Only add a label if the bar has a height greater than 0.
				plt.text(
					bar.get_x() + bar.get_width() / 2,  # X-position: center of the bar.
					height,  # Y-position: directly at the top of the bar.
					f'{int(height)}',  # Text to display: percentage formatted as an integer (no decimals).
					ha='center',  # Horizontal alignment: center the text over the bar.
					va='bottom',  # Vertical alignment: place the text just above the bar.
					color='black',  # Text color for the labels.
					fontsize=14,  # Font size of the percentage labels.
					weight='bold'  # Font weight of the percentage labels.
				)

		# Step 5: Customize the chart
		plt.xlabel('Feature', fontsize=16)  # Set the X-axis label and its font size.
		plt.ylabel('Attack Success Rate (%)', fontsize=16)  # Set the Y-axis label and its font size.

		plt.xticks(fontsize=12)  # Set font size for the X-axis tick labels.

		# Dynamically set Y-axis ticks to be suitable for percentages.
		# Determine the maximum percentage value to set the upper limit of the Y-axis.
		max_percentage = asr_percentages.max() if not asr_percentages.empty else 0
		# Set Y-ticks from 0 up to a value slightly above the maximum percentage, in steps of 10.
		# This ensures the Y-axis scale is appropriate for the data.
		y_ticks_max = max(10, int(max_percentage * 1.2))
		plt.yticks(range(0, y_ticks_max + 1, 10), fontsize=12)
		# Add horizontal grid lines for better readability of the percentage values.
		plt.grid(axis='y', linestyle='--', alpha=0.7)

		# Step 6: Save and display the chart
		plt.tight_layout()  # Adjust layout to prevent labels or elements from overlapping.
		# Save the plot to a PDF file with high DPI and a transparent background.
		plt.savefig('pre_asr_w.pdf', dpi=1000, transparent=True)
		# plt.show()  # Uncomment this line to display the plot when the script is run.


	@staticmethod
	def plot_pre_asr_g_score(file_path: Path):
		"""
		pre_score_distribution
		:param file_path:
		:return:
		"""
		df = pd.read_csv(file_path, dtype={'Harmfulness Score': int})

		plt.rcParams['font.family'] = 'serif'
		plt.rcParams['font.serif'] = ['Times New Roman']
		plt.rcParams['axes.unicode_minus'] = False

		# Step 3: Prepare data for stacked bar chart
		pivot_df = df.pivot_table(index='feature', columns='asr_g_score', values='index', aggfunc='count', fill_value=0)
		pivot_df = pivot_df.reindex(['N', 'R', 'RT'], fill_value=0)
		stacked_data = pivot_df[[1, 2, 3, 4, 5]].copy()

		# Step 4: Create the stacked bar chart
		fig, ax = plt.subplots(figsize=(4, 4))

		bar_width = 0.5
		colors = ['#B0E0E6', '#62BCC5', '#F7D96D', '#F4A674', '#D44D5B']
		bottom_values = [0] * len(stacked_data.index)

		for i, score in enumerate([1, 2, 3, 4, 5]):
			current_score_counts = stacked_data[score]
			bars = ax.bar(  # 使用 ax.bar
				stacked_data.index,
				current_score_counts,
				bottom=bottom_values,
				color=colors[i],
				label=f'{score}',
				width=bar_width
			)

			for j, bar in enumerate(bars):
				height = bar.get_height()
				if height > 0:
					x_pos = bar.get_x() + bar.get_width() / 2
					y_pos = bottom_values[j] + height / 2
					text_color = 'black'
					ax.text(  # 使用 ax.text
						x_pos,
						y_pos,
						str(int(height)),
						ha='center',
						va='center',
						color=text_color,
						fontsize=9,
						weight='bold'
					)
			bottom_values = [b + h for b, h in zip(bottom_values, current_score_counts)]

		# Step 5: Customize the chart
		ax.set_xlabel('Feature', fontsize=16)  # 使用 ax.set_xlabel
		ax.set_ylabel('Count', fontsize=16)  # 使用 ax.set_ylabel

		ax.tick_params(axis='x', labelsize=12)  # 使用 ax.tick_params
		ax.set_yticks(range(0, int(max(bottom_values)) + 1, 5))  # 使用 ax.set_yticks
		ax.tick_params(axis='y', labelsize=12)

		# 关键修改：将图例放置在 Axes 外部的顶部
		# loc='lower center'：图例自身的下部中心对齐
		# bbox_to_anchor=(0.5, 1.05)：相对于 Axes 的坐标。0.5 表示水平居中，1.05 表示 Axes 顶部边缘再往上一点
		# ncol=len(...)：使图例水平排列，适合顶部放置
		ax.legend(
			title='Harmfulness Score', loc='lower center', bbox_to_anchor=(0.5, 1.05),
			ncol=len([1, 2, 3, 4, 5]), fontsize=9, title_fontsize=14
			)

		# 添加水平网格线
		ax.grid(axis='y', linestyle='--', alpha=0.7)

		# Step 6: Save and display the chart
		plt.subplots_adjust(top=0.85)
		fig.savefig('pre_asr_g_score.pdf', dpi=1000, transparent=True, bbox_inches='tight')
		# plt.show()
