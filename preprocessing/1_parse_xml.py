import xml.etree.ElementTree as ET
import string

# 1. Incluir a tag mãe



#treat file

intab = "&�"
outtab = "   "

output = open('sports/1-aux.out', 'w', encoding = 'ISO-8859-1')

with open('sports/1-sports_all.out', 'r', encoding = "ISO-8859-1") as file:
    for line in file:
        aux = line.translate({ord(x): y for (x, y) in zip(intab, outtab)})
        aux = " ".join(aux.split())
        output.write(aux)
        output.write('\n')
output.close()




intab = ",.;/]~[´)..-!!(*&¨%$#@!<>:?}{`´'"
outtab = "                                "

intab2 = '"'
outtab2 = ' '

reviews = ET.parse('sports/1-aux.out')

fields = ['rating','title','review_text']


review = reviews.findall('review')

for field in review:
    for item in fields:
        aux = field.find(item).text
        aux = aux.translate({ord(x): y for (x, y) in zip(intab, outtab)})
        aux = aux.translate({ord(x): y for (x, y) in zip(intab2, outtab2)})
        aux = " ".join(aux.split())
        print(aux, end=' ')
    print()






# fields = ['unique_id','unique_id','asin','product_name','product_type','product_type','helpful','rating', 'title','date','reviewer','reviewer_location','review_text']
