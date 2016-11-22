import csv

infile1 = file('shuffle_CNN_validation.csv', 'rb')
reader1 = csv.reader(infile1)

infile2 = file('shuffle_CNN_training.csv', 'rb')
reader2 = csv.reader(infile2)

count = [0]*198

for label, sequence in reader1:
	count[int(label)] += 1
for label, sequence in reader2:
	count[int(label)] += 1

index = sorted(range(len(count)), key = lambda k: count[k])

for n in index:
	print('Influenza {}:'.format(n), count[n])
