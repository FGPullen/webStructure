file = open("target_cluster.results","r").readlines()
for index in range(0,6):
    score_list = []
    for i in range(index,71,6):
        line = file[i]
        score = line.strip().split()[-1]
        score = float(score)
        score_list.append(score)
    print sum(score_list)/len(score_list)
