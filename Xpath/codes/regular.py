import re

stop_label = ["input","hr","meta","link","img","br"]
re_label = re.compile(r'(?<=<).*(?=>)')
new_file = open("./toy_data/reg_android2.html","w")
lines = open("./toy_data/android2.html","r").readlines()
for line in lines:
	flag = 0
	labels = re_label.findall(line)
	for label in labels:
		tag = label.split(" ")[0]
		print tag
		if tag in stop_label:
			flag = 1
			print "delete"	
		#line = line.replace("/>","></"+tag+">")
	if flag ==0:
		new_file.write(line)
