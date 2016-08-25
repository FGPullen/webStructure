import sys, os, re





if __name__ == "__main__":

    dataset = sys.argv[1]
    cluster_id = sys.argv[2]

    num = 1000
    target_file = "../crawling/results/evaluate/target/{0}_July30_{1}_target_size5001.txt".format(dataset,cluster_id)
    lines = open(target_file,"r").readlines()

    for index,line in enumerate(lines):
        if index > 1000:
            break
        print line