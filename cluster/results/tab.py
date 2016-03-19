from tabulate import tabulate





header = ["method","micro_p","macro_p","cv_micro_p","cv_macro_p","cv_micro_filter","cv_macro_filter"]
train_lines = open("./train_batch.results","r").readlines()
cv_lines = open("./cv_batch.results","r").readlines()
result_dict = {}


for line in train_lines:
	key = ".".join(line.split("\t")[:-1])
	value = line.split("\t")[-1]
	result_dict[key] = value

for line in cv_lines:
	key = ".".join(line.split("\t")[:-1])
	value = line.split("\t")[-1]
	result_dict[key] = value

for dataset in ["stackexchange","zhihu","rottentomatoes","asp"]:
	table = []
	for method in ["kmeans","dbscan"]:
		for feat in ["tf-idf","log-tf-idf"]:
			approach = "{}({})".format(method,feat)
			element = [approach]
			for metric in ["micro_p","macro_p","cv_micro_p","cv_macro_p","cv_micro_precision","cv_macro_precision"]:
				key = "{0}.{1}.{2}.{3}".format(dataset,method,feat,metric)
				try:
					value = result_dict[key]
				except:
					value = "null"
				element.append(value)
			table.append(element)
	t = tabulate(table,header,tablefmt="latex",floatfmt=".4f")
	print t

