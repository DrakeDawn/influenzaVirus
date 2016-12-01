import csv
import random

infile = file('raw_data.csv', 'rb')
reader = csv.reader(infile)

writefile = file('shuffle_for_SOFTMAX.csv', 'ab')
writer = csv.writer(writefile)

maxlen = 0

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
					sequence = sequence.replace('A','1')
					sequence = sequence.replace('T','2')
					sequence = sequence.replace('G','3')
					sequence = sequence.replace('C','4')
					sequence = sequence.replace('U','5')

					sequence = sequence.replace('R','0')
					sequence = sequence.replace('Y','0')
					sequence = sequence.replace('K','0')
					sequence = sequence.replace('M','0')
					sequence = sequence.replace('S','0')
					sequence = sequence.replace('W','0')
					sequence = sequence.replace('B','0')
					sequence = sequence.replace('D','0')
					sequence = sequence.replace('H','0')
					sequence = sequence.replace('V','0')
					sequence = sequence.replace('N','0')
					
					
					if len(sequence) > 1900:
						sequence = sequence[:1900]
					else:
						sequence = sequence.ljust(1900, '0')
					if code > -1:
						record.append([code, information, sequence])
random.shuffle(record)
writer.writerows(record)

	#if len(sequence) > maxlen:
	#	maxlen = len(sequence)

#print(maxlen)	#2867
writefile.close()
infile.close()
