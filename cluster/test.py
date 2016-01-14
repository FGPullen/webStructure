import re 
def removeIndex(xpath):
	indexes = re.findall(r"\[\d+\]",str(xpath))
	for index in indexes:
		xpath = xpath.replace(index,"")
	return xpath


a = "div[3]"
b = "div[4]"

print removeIndex(b)