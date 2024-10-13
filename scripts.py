# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")

# Python If-Else
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n < 1 and n > 100:
        print("Number entered out of range")
    if n % 2 == 0:
        if n>=2 and n<= 5:
            print("Not Weird")
        elif n>=6 and n<= 20:
            print("Weird")
        elif n> 20:
            print("Not Weird")
    else:
        print("Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)
    print(a - b)
    print(a * b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a // b)
    print(a / b)

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(0,n):
        print(i**2)

# Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0:
        leap = True
        if year % 100 == 0:
            leap = False
            if year % 400 == 0:
                leap = True
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    result = ''
    for i in range(1,n+1):
        result += str(i)
    print(result)

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    arr = [[i, j , k] for i in range(x+1) for j in range(y+1) for k in range(z+1) if (i + j + k) != n]
    print(arr)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    arr = sorted(set(arr))
    result = arr[-2]
    print(result)

# Nested Lists
if __name__ == '__main__':
    nested = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        nested.append([name,score])
    scores = [score for name,score in nested]
    scores_set = sorted(set(scores))
    second_lowest_grade = scores_set[1]
    filtered_names = [name for name, score in nested if score == second_lowest_grade]
    if len(filtered_names) >= 2:
        filtered_names.sort()
    for name in filtered_names:
        print(name)

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    mark_sum = 0
    for mark in student_marks[query_name]:
        mark_sum+=mark
    result = mark_sum / len(student_marks[query_name])
    print(f"{result:.2f}")

# Lists
if __name__ == '__main__':
    N = int(input())
    ex = []
    for _ in range(N):
        command = input().split()
        if command[0] == 'insert':
            ex.insert(int(command[1]), int(command[2]))
        elif command[0] == 'print':
            print(ex)
        elif command[0] == 'remove':
            ex.remove(int(command[1]))
        elif command[0] == 'append':
            ex.append(int(command[1]))
        elif command[0] == 'sort':
            ex.sort()
        elif command[0] == 'pop':
            ex.pop()
        elif command[0] == 'reverse':
            ex.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    result = tuple(integer_list)
    print(hash(result))

# Introduction to Sets
def average(array):
    my_set = set(array)
    sum_of_elems = sum(my_set)
    avg = sum_of_elems / len(my_set)
    avg = round(avg, 3)
    return avg

# Symmetric Difference
M = int(input())
s1 = set(map(int, input().split()))
N = int(input())
s2 = set(map(int, input().split()))
symm_diff = s1.symmetric_difference(s2)
sorted_diff = sorted(symm_diff)
for elem in sorted_diff:
    print(elem)

# sWAP cASE
def swap_case(s):
    final_string = ''
    for i in s:
        if i.isupper():
            final_string += i.lower()
        else:
            final_string += i.upper()
    return final_string

# String Split and Join

def split_and_join(line):
    final_string = line.split(' ')
    return '-'.join(final_string)
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# What's Your Name?
#
# Complete the 'print_full_name' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. STRING first
#  2. STRING last
#
def print_full_name(first, last):
    print("Hello " + first + " " + last + "! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    my_list = list(string)
    my_list[position] = character
    final_string = ''.join(my_list)
    return final_string

# Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(0, len(string)):
        compare = string[i:len(sub_string) + i]
        if compare == sub_string:
            count += 1
    return count

# String Validators
if __name__ == '__main__':
    s = input()
    alpha = False
    alnum = False
    digit = False
    lower = False
    upper = False
    for i in s:
        if i.isalnum():
            alnum = True
        if i.isalpha():
            alpha = True
        if i.isdigit():
            digit = True
        if i.islower():
            lower = True
        if i.isupper():
            upper = True
    print(alnum)
    print(alpha)
    print(digit)
    print(lower)
    print(upper)
        

# Text Wrap

def wrap(string, max_width):
    final_string = ''
    for i in range(0, len(string), max_width):
        partial = string[i : max_width + i]
        final_string += partial + '\n'
    return final_string

# Merge the Tools!
def merge_the_tools(string, k):
    for i in range(0, len(string), k):
        check = string[i : k + i]
        unique_check = list(set(check))
        print(''.join(unique_check))

# Capitalize!

def solve(s):
    my_list = list(s)
    my_list[0] = my_list[0].upper()
    for i in range(len(my_list)):
        if my_list[i] == ' ':
            my_list[i + 1] = my_list[i + 1].upper()
    s = ''.join(my_list)
    return s

# collections.Counter()
from collections import Counter
X = int(input())
shoes_size = list(map(int,input().split()))
warehouse = Counter(shoes_size)
N = int(input())
earnings = 0
for _ in range(N):
    shoe_size, money = map(int, input().split())
    if warehouse[shoe_size] > 0:
        earnings += money
        warehouse[shoe_size] -= 1
print(earnings)

# Set .add()
n = int(input())
my_set = set()
for _ in range(1,n):
    my_set.add(str(input().split()))
print(len(my_set))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    command = input().split()
    if command[0] == 'pop':
        try:
            s.pop()
        except KeyError:
            print('Error')
    elif command[0] == 'discard':
        s.discard(int(command[1]))
    elif command[0] == 'remove':
        try:
            s.remove(int(command[1]))
        except KeyError:
            print('Error')
print(sum(s))

# Set .union() Operation
n = int(input())
A = set(map(int, input().split()))
b = int(input())
B = set(map(int, input().split()))
print(len(A.union(B)))

# Set .intersection() Operation
n = int(input())
A = set(map(int, input().split()))
b = int(input())
B = set(map(int, input().split()))
print(len(A.intersection(B)))

# Set .difference() Operation
a = int(input())
A = set(map(int, input().split()))
b = int(input())
B = set(map(int, input().split()))
print(len(A.difference(B)))

# Set .symmetric_difference() Operation
a = int(input())
A = set(map(int, input().split()))
b = int(input())
B = set(map(int, input().split()))
print(len(A.symmetric_difference(B)))

# Set Mutations
a = int(input())
A = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    command = input().split()
    B = set(map(int, input().split()))
    if command[0] == 'intersection_update':
        A.intersection_update(B)
    elif command[0] == 'update':
        A.update(B)
    elif command[0] == 'symmetric_difference_update':
        A.symmetric_difference_update(B)
    elif command[0] == 'difference_update':
        A.difference_update(B)
print(sum(A))

# Text Alignment
thickness = int(input())
c = 'H'
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
    
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Collections.namedtuple()
from collections import namedtuple
n = int(input())
columns = input().split()
marks_sum = 0
for _ in range(n):
    students_mark = namedtuple('students_mark', columns)
    MARKS, CLASS, NAME, ID = input().split()
    student = students_mark(MARKS, CLASS, NAME, ID)
    marks_sum += int(student.MARKS)    
avg = marks_sum / n
result = round(avg, 2)
print(result)

# Collections.OrderedDict()
from collections import OrderedDict
n = int(input())
my_ord_dict = OrderedDict()
for _ in range(n):
    item = input().split()
    item_name = ' '.join(item[:-1])
    net_price = int(item[-1])
    if my_ord_dict.get(item_name):
       my_ord_dict[item_name] += net_price
    else:
       my_ord_dict[item_name] = net_price
for i in my_ord_dict.keys():
    print(i, my_ord_dict[i])

# Re.split()
regex_pattern = r"[.,]"

# Group(), Groups() & Groupdict()
import re
m = re.search(r'([a-zA-Z0-9])\1+', input())
if m:
    print(m.group(1))
else:
    print(-1)

# Designer Door Mat
n, m = map(int,input().split())
for i in range(n//2):
    j = int((2*i)+1)
    print(('.|.'*j).center(m, '-'))
print('WELCOME'.center(m,'-'))
for i in reversed(range(n//2)):
    j = int((2*i)+1)
    print(('.|.'*j).center(m, '-'))

# String Formatting
def print_formatted(number):
    width = len(bin(number)[2:])
    for i in range(1, number+1):
        decimal = str(i)
        octal = oct(i)[2:]
        hexadec = hex(i)[2:].upper()
        binary = bin(i)[2:]
        print(decimal.rjust(width),octal.rjust(width),hexadec.rjust(width),binary.rjust(width))

# Calendar Module
import calendar
month, day, year = map(int, input().split())
week_day = calendar.weekday(year, month, day)
print(str(calendar.day_name[week_day]).upper())

# itertools.product()
from itertools import product
A = map(int, input().split())
B = map(int, input().split())
my_tuple = tuple(map(str, product(A,B)))
print(' '.join(my_tuple))

# Incorrect Regex
import re
n = int(input())
for _ in range(n):
    line = input()
    try:
        flag = re.compile(line)
        print(bool(flag))
    except re.error:
        print(False)

# Exceptions
n = int(input())
for _ in range(n):
    a, b = input().split()
    try:
       division = int(a) // int(b)
       print(division)
    except ZeroDivisionError as error:
        print('Error Code:', error)
    except ValueError as error:
        print('Error Code:', error)

# Arrays

def arrays(arr):
    array = numpy.array(arr, float)
    return array[::-1]
    

# Shape and Reshape
import numpy as np
my_input = list(map(int, input().split()))
my_arr = np.array(my_input)
print(np.reshape(my_arr, (3,3)))


# Transpose and Flatten
import numpy as np
N, M = map(int, input().split())
my_list = []
for _ in range(N):
    my_list.append(list(map(int, input().split())))
my_arr = np.array(my_list)
print(np.transpose(my_arr))
print(my_arr.flatten())

# Concatenate
import numpy as np
N, M , P = map(int, input().split())
A = []
for _ in range(N):
    A.append(list(map(int, input().split())))
A = np.array(A)
B = []
for _ in range(M):
    B.append(list(map(int, input().split())))
B = np.array(B)
print(np.concatenate((A, B), axis = 0))

# Zeros and Ones
import numpy as np
inp = list(map(int, input().split()))
print(np.zeros((inp), dtype = np.int64))
print(np.ones((inp), dtype = np.int64))


# Eye and Identity
import numpy as np
np.set_printoptions(legacy = '1.13')
n, m = map(int, input().split())
print(np.eye(n,m, k = 0))


# Array Mathematics
import numpy as np
n, m = map(int, input().split())
A = []
B = []
for _ in range(n):
    A.append(list(map(int, input().split())))
for _ in range(n):
    B.append(list(map(int, input().split())))
A = np.array(A)
B = np.array(B)
print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)


# Floor, Ceil and Rint
import numpy as np
np.set_printoptions(legacy = '1.13')
a = np.array(list(map(float, input().split())), dtype = float)
print(np.floor(a))
print(np.ceil(a))
print(np.rint(a))

# Sum and Prod
import numpy as np
n, m = map(int, input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))
a = np.array(a)
x = np.array(np.sum(a, axis = 0))
print(np.prod(x))


# Min and Max
import numpy as np
n, m = map(int, input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))
a = np.array(a)
min_arr = np.array(np.min(a, axis = 1))
print(np.max(min_arr))


# Mean, Var, and Std
import numpy as np
n, m = map(int ,input().split())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))
a = np.array(a)
print(np.mean(a, axis = 1))
print(np.var(a, axis = 0))
print(round(np.std(a), 11))
   


# Dot and Cross
import numpy as np
n = int(input())
a = []
for _ in range(n):
    a.append(list(map(int, input().split())))
b = []
for _ in range(n):
    b.append(list(map(int, input().split())))
a = np.array(a)
b = np.array(b)
print(np.dot(a, b))

# Inner and Outer
import numpy as np
a = np.array(list(map(int, input().split())))
b = np.array(list(map(int, input().split())))
print(np.inner(a, b))
print(np.outer(a, b))

# Linear Algebra
import numpy as np
n = int(input())
a = []
for _ in range(n):
    a.append(list(map(float, input().split())))
a = np.array(a, dtype = float)
print(round(np.linalg.det(a), 2))

# Polynomials
import numpy as np
p = np.array(list(map(float, input().split())), dtype = float)
x = int(input())
print(np.polyval(p, x))


# Alphabet Rangoli
def print_rangoli(size):
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    row = []
    for i in range(size):
        string = '-'.join(alphabet[i:size])
        row.append(string[::-1] + string[1:])
    width = len(row[0])
    
    for j in range(size-1, 0, -1):
        print(row[j].center(width, '-'))
        
    for k in range(size):
        print(row[k].center(width, '-'))
        

# The Minion Game
def minion_game(string):
    vowels = 'AEIOU'
    s_score = 0
    k_score = 0
    for i in range(len(string)):
        if string[i] in vowels:
            k_score += len(string) - i
        else:
            s_score += len(string) - i
    if k_score > s_score:
        print('Kevin' + ' ' + str(k_score))
    elif k_score == s_score:
        print('Draw')
    else:
        print('Stuart' + ' ' + str(s_score))

# DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split())
d = defaultdict(list)
for i in range(n):
    elem = input()
    d[elem].append(i + 1)
for j in range(m):
    elemB = input()
    if elemB in d:
        print(*d[elemB])
    else:
        print(-1)

# No Idea!
n, m = map(int, input().split())
my_list = input().split()
a = set(input().split())
b = set(input().split())
happiness = 0
for item in my_list:
    if item in a:
        happiness += 1
    elif item in b:
        happiness -= 1
print(happiness)

# The Captain's Room
k = int(input())
my_list = input().split()
my_set = set(my_list)
for elem in my_set:
    my_list.remove(elem)
cap_room = my_set.difference(set(my_list))
print(*cap_room)

# Check Subset
t = int(input())
for _ in range(t):
    n = int(input())
    a = set(input().split())
    m = int(input())
    b = set(input().split())
    print(a.issubset(b))

# Check Strict Superset
a = set(input().split())
n = int(input())
superset = True
for _ in range(n):
    b = set(input().split())
    if a.issuperset(b) != True or len(a) == len(b):
        print(False)
        break
else:
    print(True)

# Collections.deque()
from collections import deque
d = deque()
n = int(input())
for _ in range(n):
    command = input().split()
    if command[0] == 'append':
        d.append(command[1])
    if command[0] == 'pop':
        d.pop()
    if command[0] == 'popleft':
        d.popleft()
    if command[0] == 'appendleft':
        d.appendleft(command[1])
print(*d)

# Company Logo
#!/bin/python3
import math
import os
import random
import re
import sys
from collections import Counter

if __name__ == '__main__':
    s = input()
    s = sorted(s)
    count = Counter(s)
    for key, value in count.most_common(3):
        print(key, value)

# Piling Up!
t = int(input())
for _ in range(t):
    n = int(input())
    blocks = list(map(int, input().split()))
    for i in range(n - 1):
        if blocks[0] >= blocks[len(blocks)-1]:
            bottom = blocks[0]
            blocks.pop(0)
        elif blocks[len(blocks) - 1] > blocks[0]:
            bottom = blocks[len(blocks)-1]
            blocks.pop(len(blocks)-1)
        else:
            pass
        if len(blocks) == 1:
            print('Yes')
        if blocks[0] > bottom or blocks[len(blocks)-1] > bottom:
            print('No')
            break

# Word Order
n = int(input())
my_dict = {}
for _ in range(n):
    word = str(input())
    if word in my_dict:
        my_dict[word] += 1
    else:
        my_dict[word] = 1
print(len(my_dict))
values = map(str, my_dict.values())
print(' '.join(values))

# Time Delta
#!/bin/python3
import math
import os
import random
import re
import sys
from datetime import datetime
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    date1 = datetime.strptime(t1, time_format)
    date2 = datetime.strptime(t2, time_format)
    return str(int(abs((date1-date2).total_seconds())))
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        t1 = input()
        t2 = input()
        delta = time_delta(t1, t2)
        fptr.write(delta + '\n')
    fptr.close()

# Zipped!
n, x = map(int, input().split())
my_list = []
for _ in range(x):
    my_list.append(list(map(float, input().split())))
students_mark = zip(*my_list)
for elem in students_mark:
    print(round(sum(elem) / x, 1))

# Decorators 2 - Name Directory

from operator import itemgetter
def person_lister(f):
    def inner(people):
        people = sorted(people, key=lambda x: int(x[2]))
        return [f(person) for person in people]
    return inner
    

# Athlete Sort
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()
    n = int(nm[0])
    m = int(nm[1])
    arr = []
    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))
    k = int(input())
    arr = sorted(arr, key = lambda x: int(x[k]))
    for elem in arr:
        print(*elem)

# ginortS
string = str(input())
lower = sorted([i for i in string if i.islower()])
upper = sorted([i for i in string if i.isupper()])
odd = sorted([i for i in string if i.isdigit() and int(i) % 2 != 0])
even = sorted([i for i in string if i.isdigit() and int(i) % 2 == 0])
print(''.join(lower + upper + odd + even))

# Map and Lambda Function
cube = lambda x: x**3
def fibonacci(n):
    fib_seq = [0, 1]
    list(map(lambda _: fib_seq.append(fib_seq[-1] + fib_seq[-2]), range(2, n)))
    return fib_seq[:n]

# Re.findall() & Re.finditer()
import re
vowels = 'AEIOUaeiou'
consonants = 'QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm'
my_regex = re.findall(r'(?<=[' + consonants + '])([' + vowels + ']{2,})(?=[' + consonants + '])', input())
if my_regex:
    for pattern in my_regex:
        print(pattern)
else:
    print(-1)

# Detect Floating Point Number
import re
t = int(input())
for _ in range(t):
    match = re.match(r'^[-+]?[0-9]*\.[0-9]+$', input())
    if match:
        print(True)
    else:
        print(False)

# Regex Substitution
import re
n = int(input())
for _ in range(n):
    inp = input()
    inp = re.sub(r'(?<=\s)&&(?=\s)', 'and', inp)
    inp = re.sub(r'(?<=\s)\|\|(?=\s)', 'or', inp)
    print(inp)

# Validating Roman Numerals
regex_pattern = r"^M{0,3}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"

# Birthday Cake Candles
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#
def birthdayCakeCandles(candles):
    max_count = 0
    max_height = max(candles)
    for i in candles:
        if i == max_height:
            max_count += 1
    return max_count
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    candles_count = int(input().strip())
    candles = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(candles)
    fptr.write(str(result) + '\n')
    fptr.close()

# Insertion Sort - Part 1
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort1(n, arr):
    elem = arr[-1]
    for i in range(n-1, 0, -1):
        if elem < arr[i-1]:
            arr[i] = arr[i-1]
            print(*arr)
        else:
            arr[i] = elem
            print(*arr)
            break
    else:
        arr[0] = elem
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#
def insertionSort2(n, arr):
    for i in range(1, n):
        elem = arr[i]
        j = i-1
        while j >= 0 and arr[j] > elem:
            arr[j+1] = arr[j]
            j -=1
        arr[j+1] = elem
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)

# Number Line Jumps
#!/bin/python3
import math
import os
import random
import re
import sys
import itertools
#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#
def kangaroo(x1, v1, x2, v2):
    for i in range(10000):
        if (x1 + (v1 * i)) == (x2 + (v2 * i)):
            return 'YES'
    else:
        return 'NO'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    x1 = int(first_multiple_input[0])
    v1 = int(first_multiple_input[1])
    x2 = int(first_multiple_input[2])
    v2 = int(first_multiple_input[3])
    result = kangaroo(x1, v1, x2, v2)
    fptr.write(result + '\n')
    fptr.close()

# Recursive Digit Sum
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#
def superDigit(n, k):
    n = sum([int(n[digit]) for digit in range(len(n))]) * k
    return superDigit(str(n), 1) if n > 9 else n
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    first_multiple_input = input().rstrip().split()
    n = first_multiple_input[0]
    k = int(first_multiple_input[1])
    result = superDigit(n, k)
    fptr.write(str(result) + '\n')
    fptr.close()

# Validating phone numbers
import re
n = int(input())
for _ in range(n):
    number = input()
    regex = re.match(r'[789]\d{9}$',  number)
    if regex:
        print('YES')
    else:
        print('NO')

# Validating and Parsing Email Addresses
import re
import email.utils 
N = int(input())
regex = r'^[a-z][\w\-\.]+@[a-z]+\.[a-z]{1,3}$'
for i in range(N):
    parsed_input = email.utils.parseaddr(input())
    if re.search(regex, parsed_input[1]):
        print(email.utils.formataddr(parsed_input)) 

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        result = []
        for i in l:
            number = str('+91 ' + i[-10:-5] + ' ' + i[-5:])
            result.append(number)
        f(result)
    return fun

# Viral Advertising
#!/bin/python3
import math
import os
import random
import re
import sys
#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#
def viralAdvertising(n):
    shares = 5
    likes = shares // 2
    for i in range(n-1):
        like_per_day = shares // 2
        shares = like_per_day * 3
        likes += shares // 2
    return likes
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    n = int(input().strip())
    result = viralAdvertising(n)
    fptr.write(str(result) + '\n')
    fptr.close()

# Hex Color Code
import re
N = int(input())
for _ in range(N):
    line = input()
    regex = re.findall(r"(\#[a-f0-9]{3,6})[\;\,\)]{1}", line, re.IGNORECASE)
    if regex:
        for j in list(regex):
            print(j)

# Validating UID
import re
n = int(input())
for _ in range(n):
    uid = str(input())
    regex = re.match(r'^(?=(?:.*[A-Z]){2})(?=(?:.*\d){3})[A-Za-z0-9]{10}$', uid)
    if (regex):
        if len(set(uid)) == len(uid):
            print('Valid')
        else:
            print('Invalid')
    else:
        print('Invalid')

# Re.start() & Re.end()
import re
string = input()
substring = input()
regex = list(re.finditer(f'(?={substring})', string))
if regex:
    for i in regex:
        print((i.start(), i.start() + len(substring) - 1))
else:
    print((-1, -1))
    

# XML 1 - Find the Score

def get_attr_number(node):
    return len(node.attrib) + sum(get_attr_number(child) for child in node)

# XML2 - Find the Maximum Depth

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for child in elem:
        depth(child, level + 1)

# HTML Parser - Part 1
from html.parser import HTMLParser
N = int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print('Start :', tag)
        for elem in attrs:
            print('->', elem[0], '>', elem[1])
            
    def handle_endtag(self, tag):
        print('End   :', tag)
        
    def handle_startendtag(self, tag, attrs):
        print('Empty :', tag)
        for elem in attrs:
            print('->', elem[0], '>', elem[1])
            
parser = MyHTMLParser()
html_string = ''.join([input().strip() for i in range(0, N)]) 
parser.feed(html_string)

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
N = int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        [print('-> {} > {}'.format(*attr)) for attr in attrs]
    
html = '\n'.join([input() for x in range(0, N)])
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating Credit Card Numbers
import re
N = int(input())
regex_pattern = []
structure_re = r'[456]\d{3}(-?\d{4}){3}$'
no_consecutive_re = r'((\d)-?(?!(-?\2){3})){16}'
regex_pattern.append(structure_re)
regex_pattern.append(no_consecutive_re)
for _ in range(N):
    credit_card = input()
    if re.match(regex_pattern[0], credit_card):
        if re.match(regex_pattern[1], credit_card):
            print('Valid')
        else:
            print('Invalid')
    else:
        print('Invalid')

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if (data.find('\n') != -1):
            print('>>> Multi-line Comment')
        else:
            print('>>> Single-line Comment')
        print(data)
        
    def handle_data(self, data):
        if data == '\n':
            return
        print('>>> Data')
        print(data)
  
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating Postal Codes
regex_integer_in_range = r"^[1-9]\d{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"

# Matrix Script
#!/bin/python3
import math
import os
import random
import re
import sys
first_multiple_input = input().rstrip().split()
n = int(first_multiple_input[0])
m = int(first_multiple_input[1])
matrix = []
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
my_list = []
for i in range(m):
    for e in range(n):
        my_list.append(matrix[e][i])
string = ''.join(my_list)
pattern = r"([a-zA-Z0-9])([^a-zA-Z0-9]+)([a-zA-Z0-9])"
def replace_special_characters(match):
    return f"{match.group(1)} {' '}{match.group(3)}"
result = re.sub(pattern, replace_special_characters, string)
print(result)

