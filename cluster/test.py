import collections
def get_cluster_number_shift(labels_true, labels_pred):
	true_set = set(labels_true)
	pre_set = set(labels_pred)
	dic = {} 
	for item in pre_set:
		dic[item] = {}
		for item_2 in true_set:
			dic[item][item_2] = 0

	for i in range(len(labels_true)):
		dic[labels_pred[i]][labels_true[i]] += 1
	print dic		
	final_dict = collections.defaultdict(dict)
	used_list = set()
	for pred_key in pre_set:
		max_value = -1
		print dic[pred_key]
		for index, value in dic[pred_key].iteritems():
			if index not in used_list:
				if value > max_value:
					max_label = index
					max_value = value
			final_dict[pred_key] = max_label
			used_list.add(max_label)
	return final_dict


true = [1, 1, 1, 1, 1, 2, 2, 3, 3]
pred= [2, 2, 2, 3, 3, 3, 4, 4, 4]

dic = get_cluster_number_shift(true, pred)
print dic
print dic
print dic
