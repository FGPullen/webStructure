from lxml import etree
from io import StringIO, BytesIO
html = open("data.html","r")
parser = etree.HTMLParser()
tree = etree.parse(html,parser)
result = etree.tostring(tree.getroot(),pretty_print=True, method="html")
print (result)
