import os 
import re


if __name__ == "__main__":
	folder_name = "../Crawler/test_data/ASP/"
	'''
	re_list = re.compile(r'(?<=http://forums.asp.net/)([/0-9]+?)(?=.aspx/)')
	for file_path in os.listdir(folder_name):
		path = file_path.replace("_","/")
		if "/f/" in path:
			tag = 2
		elif "/members/" in path:
		    tag = 0
		elif "RedirectToLogin" in path or "/private-message/" in path:
		    tag = 1
		elif "/post/" in path:
		    tag = 3
		elif "/t/" in path or "/p/" in path:
		    tag =3
		elif "search?" in path:
		    tag =4
		else:
			tag = 5
			match = re_list.search(path)
			if not match:
				print path
				os.remove(folder_name+file_path)
	'''
	index_file = open("./Data/asp_wrong_page.txt","r").readlines()
	for index,file in enumerate(index_file):
		index_file[index] = file.strip().replace("/","_")
	print len(index_file)
	num = 0
	for file_path in os.listdir(folder_name):
		path = file_path.replace(".html","")
		if path in index_file:
			print file_path
			os.remove(folder_name+file_path)
			num += 1
	print num