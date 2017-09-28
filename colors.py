from csv import reader
from pprint import PrettyPrinter

pp = PrettyPrinter()
with open('satfaces.csv', newline='') as sat:
    next(sat)
    colors = set()
    satread = reader(sat, delimiter=',')
    for i in satread:
        colors.add(i[3])
    print("{} different colors".format(len(colors)))
    pp.pprint(colors)
