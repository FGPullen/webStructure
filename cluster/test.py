import collections
from lxml import etree

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


def getXpaths(self,index=False):
    # TODO: XPaths are pretty deep and it becomes noisier
    # when it goes deeper.  Pruning might be a good idea.
    tree= etree.HTML(str(self.contents))
    Etree = etree.ElementTree(tree)
    nodes = tree.xpath("//*[not(*)]")
    for node in nodes:
        # we do not consider index or predicate here
        xpath = Etree.getpath(node)
        #self.dfs_xpaths_list.append(xpath) # except for this one
        if not index:
            xpath = self.removeIndex(xpath)
        #xpath = "/".join(xpath.split('/')[:-1]) # prune the leaf level
        #xpath = self.stemming(xpath)
        self.dfs_xpaths_list.append(xpath)
        self.addXpath(xpath)

def getDFSXpaths(self, root, xpath=""):
    loop_node = self.detect_loop(root)
    for node in root:
        if type(node.tag) is not str:
            continue
        if loop_node:
            new_xpath = "/".join([xpath, node.tag, 'loop'])
        else:
            new_xpath = "/".join([xpath, node.tag])
        if len(node) == 0:
            #print new_xpath
            self.dfs_xpaths_list.append(new_xpath)
            self.addXpath(new_xpath)
        if len(node) != 0:
            self.getDFSXpaths(node, new_xpath)
        if loop_node:
            break

if __name__ == "__main__":
	
