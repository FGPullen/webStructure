import os
import sys
from urllib.parse import urlparse
import random
import lxml.html
import warc
import time

if __name__ == '__main__':
    site = sys.argv[1]
    site_dir = sys.argv[2]
    arc_file = os.path.join(site_dir, "{0}.arc.gz".format(site))
    prefix = "http://www.{0}.com".format(site)
    #write_file = open("{0}.link.txt".format(site),"w")

    f = warc.open(arc_file)
    record_num = 0
    urlset = set()
    for record in f:
        url = record['URL']
        urlset.add(url)
        record_num += 1

    f = warc.open(arc_file)
    edge_count = 0
    edge_total = 0
    processed = 0
    unexist_set = set()
    for record in f:
        inlink_total,inlink_count = 0,0
        inlink_set = set()
        url = record['URL']
        if url != "http://www.rottentomatoes.com/":
            continue
        current_dir = '/'.join(url.split('/')[:-1])
        content = record.payload.read()
        try:
            root = lxml.html.fromstring(content)
            hrefs = root.xpath('//a/@href')
            for href in hrefs:
                href_str = str(href).strip()
                if not href_str:
                    continue
                if href_str[0] == '#':
                    continue
                if href_str.startswith('javascript:'):
                    continue
                if href_str.startswith('mailto:'):
                    continue
                if href_str[0] == '/':
                    if len(href_str) > 1 and href_str[1] == '/':
                        continue
                    href_str = prefix + href_str

                if not href_str.startswith('http'):
                    href_str = current_dir + href_str
                parsed_uri = urlparse(href_str)
                domain = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)
                if not site in domain:
                    continue
                if href_str in inlink_set:
                    continue
                else:
                    inlink_set.add(href_str)
                edge_total += 1
                inlink_total += 1
                if href_str in urlset:
                    edge_count += 1
                    inlink_count += 1
                else:
                    print(href_str)
                    unexist_set.add(href_str)
        except:
            pass
        #print (str(inlink_count) + "\t" + str(inlink_total))
        #write_file.write(url+"\t"+str(inlink_count) + "\t" + str(inlink_total)+"\n")
        processed += 1
        if processed % 100 == 0:
            sys.stderr.write("Processed {0}/{1}\n".format(processed, record_num))

    vertex_num = len(urlset)

    print("Unexist_set size: {0}".format(len(unexist_set)))
    print("Vertex Number: {0}".format(vertex_num))
    print("Edge Number: {0}".format(edge_count))
    print("Link Coverage: {0}".format(float(edge_count) / edge_total))
    print("Graph Density: {0}".format(float(edge_count) / vertex_num / (vertex_num - 1)))