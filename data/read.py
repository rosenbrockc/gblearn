def read():
    pdict = {}
    iskip = 0;
    with open("mobility.csv") as f:
        for line in f:
            if iskip < 1:
                    iskip += 1
                    continue
            rvals = line.split(",")
            gbid = rvals[0]
            pval = int(rvals[8])
            pdict[gbid] = pval
    return pdict

