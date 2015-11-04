class cluster:
	def __init__(self):
		self.pages = []
		self.IntraD = 0.0

	def addPage(self,page_object):
		self.pages.append(page_object)

	#def cal_IntraD(self):
