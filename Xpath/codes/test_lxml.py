from io import StringIO, BytesIO
from lxml import etree


#string = open('../../cluster/test/data.html',"r").readlines()
string = open('./toy_data/android2.html',"r").read().replace('\n','')
print type(str(string))
tree = etree.HTML(str(string))
result =  etree.tostring(tree, pretty_print = True, method="html")

#a = etree.Element()
r = tree.xpath("//div[@class='topbar']")

print str(len(r))
#print str(len(r))+"\t"+str(type(r[0]))
#print tree.getpath(r[0])

result = result.replace(r"\r\n","")
#print result
outputFile = open("clean_html.html","w")
outputFile.write(result)
