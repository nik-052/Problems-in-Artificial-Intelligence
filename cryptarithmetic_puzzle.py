import itertools
first = input('Enter the first string: ')
one = first
second = input('Enter the second string: ')
two = second
ans = input('Enter the answer string: ')
answer = ans
numbers = range(10)
first = [i for i in first]
second = [i for i in second]
ans = [i for i in ans]
letters = []
for item in first:
    letters.append(item)
for item in second:
    letters.append(item)
for item in ans:
    letters.append(item)
letters = list(set(letters))
for p in itertools.permutations(numbers, len(letters)):
    s = dict(zip(letters,p))
    if s[first[0]]==0 or s[second[0]]==0:
        continue
    send = 1000 * s[first[0]] + 100 * s[first[1]] + 10 * s[first[2]] + s[first[3]]
    more = 1000 * s[second[0]] + 100 * s[second[1]] + 10 * s[second[2]] + s[second[3]]
    money = 10000 * s[ans[0]] + 1000 * s[ans[1]] + 100 * s[ans[2]] + 10 * s[ans[3]] + s[ans[4]]        
    if send + more == money:
        print(one,': ',send)
        print(two,': ',more)
        print(answer,': ',money)
