import csv

infile = open("influenza.fna")
csvfile = file("raw_data.csv", "wb")
writer = csv.writer(csvfile)

label = '>label'
sequence = 'sequence'
count = 0

for line in infile.readlines():
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
