# -*- coding: utf-8 -*-
import csv
import math
import numpy as np 

datafile='src/data/DAT_MT_EURUSD_M1_2019.csv'
intervalSize=20

data1=list()

with open(datafile) as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')
	#line_count = 0
	for row in csv_reader:
		data1.append(row[3:6])
		'''
		line_count += 1
		if line_count == 100:
			break
		#'''

rowsOriginal=len(data1)-1
columns=data1[0]
data2=data1[1:len(data1)]

intervals=len(data2)
intervals=math.floor(intervals/intervalSize)
rows=intervals*intervalSize

data2=data2[0:rows]

'''
print(rowsOriginal)
print(rows)
print(intervals)
print(columns)
#print(data1)
#'''

data3=list()
for c in range(intervals):
	data3.append(data2[intervalSize*c:intervalSize*(c+1)])
data3=np.array(data3,np.float)

close=data3[0:len(data3)-1,intervalSize-1,2]
close2=close.repeat(60)
close2=close2.reshape(len(data3)-1,intervalSize,3)

close=close.repeat(3)
close=close.reshape(len(data3)-1,3)

train_data=data3[0:intervals-1]-close2
next_labels=data3[1:len(data3),0]-close
avg_labels=np.average(data3[1:len(data3)],1)-close
train_labels=np.concatenate((next_labels,avg_labels),axis=1)

del c
del data1
del data2
del data3
del rowsOriginal
del rows
del row
del intervals
del csv_file
del csv_reader
del next_labels
del avg_labels
