
def readData(fileName):
    dataDict = dict()
    edges = []
    with open(fileName, 'r') as f:
        allData = f.readlines()
        for data in allData[:8]:
            data = data.split(":")
            dataDict[data[0].strip()] = data[1].strip()
        for data in allData[8:]:
            if 'END' in data:
                break
            if 'NODES' in data:
                continue
            data = data.split()
            edges.append((int(data[0]), int(data[1]), int(data[2]), int(data[3])))
        return dataDict, edges

readData("sample.dat")