from tabulate import tabulate
from collections import defaultdict


header = ["micro_f", "macro_f", "micro_p","macro_p","cv_micro_precision","cv_macro_precision"]

train_lines = open("./c_train_baseline.results","r").readlines()
cv_lines = open("./c_cv_baseline.results","r").readlines()
result_dict = defaultdict(list)

for line in train_lines:
	key = ".".join(line.split("\t")[:-1])
	#if len(line.split("\t")) <= 2:
	# 		continue
	line = line.strip()
	print line, len(line.split("\t"))
	#raise
	value = line.split("\t")[-1]
	metric = line.split("\t")[-2]
	for key in header:
		if key == metric:
			result_dict[key].append(float(value))

for line in cv_lines:
	key = ".".join(line.split("\t")[:-1])
	if len(line.split("\t")) <= 2:
		continue
	line = line.strip()
	value = line.split("\t")[-1]
	metric = line.split("\t")[-2]
	for key in header:
		if key == metric:
			result_dict[key].append(float(value))

print result_dict

for key,value in result_dict.iteritems():
	print key,value