
enst = set()
enst2gene = {}
with open('gene_mutation_rate0520.txt') as f:
    print f.readline()
    for line in f:
        lst = line.strip().split()
        enst.add(lst[3])
        enst2gene[lst[3]] = lst[0]
        
with open('mis_rate_hongjian0526.txt') as f, open('mis_rate_hongjian_onegene_0526.txt', 'w') as fw:
    head = f.readline().strip().split()
    head[0] = 'Gene'
    fw.write('\t'.join(head) + '\n')
    for line in f:
        lst = line.strip().split()
        if lst[1] in enst:
            lst[0] = enst2gene[lst[1]]
            fw.write('\t'.join(lst) + '\n')