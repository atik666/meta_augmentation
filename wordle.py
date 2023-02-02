import string
alphabet = list(string.ascii_lowercase)
print(alphabet)

remove = ['l','a','t','e','p','u','n','m','o','c','z','x','q','j']

new = [x for x in alphabet if x not in remove]

from itertools import combinations

com_set = combinations(new, 3)

x = list(combinations(new, 3)) 

aa = []
for i in x:
    if 'i' in i:
        aa.append(i)
        
