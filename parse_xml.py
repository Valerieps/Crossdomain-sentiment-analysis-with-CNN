import xml.etree.ElementTree as ET
import string

#treat file

intab = "&"
outtab = " "

output = open('data/sorted_data/apparel/all_treated.review', 'w')

with open('data/sorted_data/apparel/all.review', 'r') as file:
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

# reviews = ET.parse('data/sorted_data/apparel/test_xml.review')
reviews = ET.parse('data/sorted_data/apparel/all_treated.review')

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
