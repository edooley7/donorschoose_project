from __future__ import division
import csv
count = 0
with open('opendata_essays.csv', 'rb') as infile, open('cleaned_essays.csv', 'wb') as outfile:
    reader = csv.reader(infile, delimiter=',', quotechar='"', escapechar = '\\')
    writer = csv.writer(outfile, delimiter=',', quotechar='"', escapechar = '\\')
    for row in reader:
        row = [r or "NULL" for r in row]
        count +=1
        print count/878000
        writer.writerow(row)