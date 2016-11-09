import csv

infile = open("influenza.fna")
csvfile = file("csv_test.csv", "wb")
writer = csv.writer(csvfile)

label = '>label'
sequence = 'sequence'
count = 0

for line in infile.readlines():
	count += 1
	if count % 1000 == 0:
		print(count / 1000)
	if line[0] == '>':
		writer.writerow([label,sequence])
		label = line
		label = label.rstrip()
		label = label.replace(',', ' ')
		label = label.replace(';', ' ')
		sequence = ''
	else:
		sequence += line
		sequence = sequence.rstrip()
writer.writerow([label,sequence])

infile.close()
csvfile.close()