from ml_deeco.utils import Log


class AverageLog(Log):

    def add(self, newData):
        currentRecord = self.records[-1]
        for i in range(len(currentRecord)):
            currentRecord[i] = currentRecord[i] + newData[i]
        self.records[-1] = currentRecord

    def average(self, begin=0, end=1):
        assert begin < end, "begining index must be less than ending index"
        # ignoring first column
        datalist = self.records[begin:end]
        count = end - begin
        averageList = [sum([sublist[i] for sublist in datalist]) / count for i in range(self.columns)]
        return averageList

    def totalRecord(self):
        datalist = self.records
        count = len(datalist)
        if count == 0:
            return []
        averageList = [sum([sublist[i] for sublist in datalist]) / count for i in range(self.columns)]
        return averageList
