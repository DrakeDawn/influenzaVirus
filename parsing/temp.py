import csv

infile = file("HON1.csv", 'rb')
outfile = file("HON1_1.csv", "wb")

reader = csv.reader(infile)
writer = csv.writer(outfile)

for label, info, sequence in reader:
	sequence = sequence.replace('A','1')
	sequence = sequence.replace('T','2')
	sequence = sequence.replace('G','3')
	sequence = sequence.replace('C','4')
	sequence = sequence.replace('U','5')
	if len(sequence) > 1900:
		sequence = sequence[:1900]
	else:
		sequence = sequence.ljust(1900, '0')
	writer.writerow([label,info,sequence])

infile.close()
outfile.close()