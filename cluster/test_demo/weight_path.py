xpaths= open("../Files/weight.txt","r").readlines()[0].strip().split("\t")
dic = []
for xpath in xpaths:
	if xpath not in dic:
		dic.append(xpath)

path_list = ["/html/body/div/div/div/link","/html/body/div/div/div/div/div/ul/li/div","/html/body/div/div/div/div/div/table/tr/td/div/div/b","/html/body/div/div/div/div/div/table/tr/td/div/table/tr/td/div/div/br","/html/body/div/div/div/div/div/table/tr/td/div/input"]
path_index = []
for path in path_list:
	path_index.append([k for (k, v) in enumerate(dic) if v == path])
print path_index

weight_file = open("../Files/values.txt","r").readlines()
for index, line in enumerate(weight_file):
	if index != 1:
		continue
	weights = line.strip().split("\t")
	for path in path_index:
		path = path[0]
		print weights[path]
	weights_dict = {}
	for key,value in enumerate(weights):
		weights_dict[key] = value
	sort_list = sorted(weights_dict.iteritems(), key=lambda d:d[1], reverse = True)
	#print str(sort_list[0][0]) + "\t" + str(sort_list[0][1])
	#key = sort_list[0][0]
	#print dic[key]
	#sort_list[]