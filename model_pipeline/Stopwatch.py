import pandas as pd
from timeit import default_timer as timer

class Stopwatch:
	def __init__(self):
		self.timer = []
		self.add('start')

	def add(self, name):
		self.timer.append({
				'name': name,
				'time': timer()
			})
		if len(self.timer) > 1:
			self.timer[-1]['elapsed'] = self.timer[-1]['time'] - self.timer[-2]['time']
		else:
			self.timer[-1]['elapsed'] = 0

	def write(self, file_path):
		pd.DataFrame(self.timer).to_csv(file_path, index=False)