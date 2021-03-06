import csv
import random
import math

infile = file('raw_data.csv', 'rb')
reader = csv.reader(infile)

writefile = file('shuffle_2.csv', 'ab')
writer = csv.writer(writefile)

maxlen = 0
count = 0
count2 = 0


record = []

for label, sequence in reader:
	if label[0] == '>':
		if label.find('Influenza A virus') > -1:
			index = label.find('))')
			indexH = label.find('H', index - 6, index)
			indexN = label.find('N', indexH + 1, index)
			if index > -1 and indexH > -1 and indexN > -1:
				H = label[(indexH + 1):indexN]
				N = label[(indexN + 1):index]
				if H.isdigit() and N.isdigit():
					H = int(H)
					N = int(N)
					code = (H - 1) * 11 + N - 1
					information = label[(index + 3):]
					m = int(math.ceil(len(sequence) / 1000.0))
					maxlen += len(sequence)
					count+=1

					sequence = sequence.replace('A','1000')
					sequence = sequence.replace('T','0100')
					sequence = sequence.replace('G','0010')
					sequence = sequence.replace('C','0001')
					sequence = sequence.replace('U','0100')

					sequence = sequence.replace('R','0000')
					sequence = sequence.replace('Y','0000')
					sequence = sequence.replace('K','0000')
					sequence = sequence.replace('M','0000')
					sequence = sequence.replace('S','0000')
					sequence = sequence.replace('W','0000')
					sequence = sequence.replace('B','0000')
					sequence = sequence.replace('D','0000')
					sequence = sequence.replace('H','0000')
					sequence = sequence.replace('V','0000')
					sequence = sequence.replace('N','0000')

					sequence = sequence.ljust(m * 4000, '0')

					if code > -1:
						for i in range(m):
							record.append([code, sequence[(4000 * i):(4000 * (i + 1))]])
							count2 += 1
						
random.shuffle(record)
writer.writerows(record)

print(maxlen/float(count))
print(count2)

	#if len(sequence) > maxlen:
	#	maxlen = len(sequence)

#print(maxlen)	#2867
writefile.close()
infile.close()
