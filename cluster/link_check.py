from pages import allPages

class link_analyzer:
	def __init__(self, data_path,dataset):
		self.pages = allPages([data_path])
		self.dataset = dataset
		prefix = data_path 
		self.file_set = []
		for page in self.pages.pages:
			self.file_set.append(page.path)
		#print file_set

	def getAnchor(self):
		self.right_list = []
		self.total_list = []
		self.percentage_list = []
		for page in self.pages.pages:
			right = 0
			total = 0
			link_dict = page.getAnchor()
			for key,link in link_dict.iteritems():
				if self.intraJudge(link):
					for item in self.file_set:
						if link in item:
							right += 1
							print link
							break
					total += 1
			if right ==0:
				print 0.0
				self.percentage_list.append(0.0)
			else:
				link_dict["percentage"] = float(right)/float(total)
				self.percentage_list.append(float(right)/float(total))
			self.right_list.append(right)
			self.total_list.append(total)
		print "average percentage is " + str(sum(self.percentage_list)/float(len(self.percentage_list)))
		print "average inlink number is " + str(sum(self.total_list)/float(len(self.total_list)))



	def intraJudge(self,url):
		# oulink with http or symbol like # and /
		# medhelp start from http://www.medhelp.org/user_groups/list and prefix http://www.medhelp.org/
		if self.dataset == "stackexchange":
			if "http" in url:
				return 0
			elif "//" in url:
				return 0
			elif url=="#" or url=="?lastactivity":
				return 0
			else:
				return 1
		elif self.dataset == "rottentomatoes":
			if len(url)==1 or "http" in url:
				if "rottentomatoes.com" in url:
					return 1
				else:
					return 0
			elif url[0:2]=="//":
				return 0
			else:
				return 1

if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser()
	parser.add_argument("datasets", choices=["zhihu","stackexchange","rottentomatoes","medhelp","asp","all"], help="the dataset for experiments")
	args = parser.parse_args()
	if args.datasets!="all":
		data_path = "../Crawler/test_data/" + args.datasets + "/"
		l = link_analyzer(data_path,args.datasets)
		l.getAnchor()
	else:
		for data in ["zhihu","stackexchange","rottentomatoes","medhelp","asp"]:
			data_path  = "../Crawler/test_data/" + data + "/"
			l  = link_analyzer(data_path)
