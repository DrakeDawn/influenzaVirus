import csv

threshold = 10

infile1 = file('shuffle_CNN_validation.csv', 'rb')
reader1 = csv.reader(infile1)

infile2 = file('shuffle_CNN_training.csv', 'rb')
reader2 = csv.reader(infile2)

outfile = file('renaming.csv', 'wb')
writer = csv.writer(outfile)

count = [0]*198

for label, sequence in reader1:
	count[int(label)] += 1
for label, sequence in reader2:
	count[int(label)] += 1

index = sorted(range(len(count)), key = lambda k: count[k], reverse = True)

for n in range(198):
	if count[index[n]] > threshold:
		writer.writerow(['{}'.format(index[n]), '{}'.format(n)])
