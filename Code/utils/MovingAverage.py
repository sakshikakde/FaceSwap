import numpy as np

class MovingAverage:

    def __init__(self, window_size, weight):

        self.window_size_ = window_size
        self.markers_ = []
        self.average_ = 0
        self.weight_ = weight

    def addMarkers(self, points):

        if len(self.markers_) < self.window_size_:
            self.markers_.append(points)

        else:
            self.markers_.pop(0)
            self.markers_.append(points)

    def getAverage(self):
        markers = self.markers_
        markers = np.array(markers)
        sum = np.sum(markers, axis = 0)
        # sum = 0
        # for i in range(len(self.markers_)):
        #     sum = sum + self.weight_[i] * markers[i,:]

        # self.average_ = (sum / np.sum(self.weight_)).astype(int)
        self.average_ = (sum / len(self.markers_)).astype(int)
        if len(self.markers_) < self.window_size_:
            return markers[-1]
        else:
            return self.average_

    def getListLength(self):
        l = len(self.markers_)
        return l





