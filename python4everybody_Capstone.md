What is the most common Unicode encoding when moving data between systems?
ans: UTF-8

What is the decimal (Base-10) numeric value for the upper case letter "G" in the ASCII character set?
ans: 71

What word does the following sequence of numbers represent in ASCII:
108, 105, 115, 116
ans: list


```python
How are strings stored internally in Python 3?
ans: unicode
```


```python
When reading data across the network (i.e. from a URL) in Python 3, what method must be used to convert it to the internal format used by strings? 
ans: decode
```


```python
if x < 2 :
    print("Below 2")
elif x < 0 :
    print("Negative")
else :
    print("Something else")
ans: Negative
```

if x < 2 :
    print("Below 2")
elif x < 20 :
    print("Below 20")
elif x < 10 :
    print("Below 10")
else :
    print("Something else")
ans: below 10


```python
stuff = ['joseph', 'sally', 'walter', 'tim']
print(stuff[2])
```

    walter
    


```python

```


```python
def hello():
    print("Hello")
    print("There")

x = 10
x = x + 1
```


```python
x = -1
for value in [3, 41, 12, 9, 74, 15] :
    if value > x :
        x = value
print(x)
```

    74
    


```python
total = 0
for abc in range(5):
    total = total + abc
print(total)
```

    10
    


```python
def fred():
   print("Zap")

def jane():
   print("ABC")

jane()
fred()
jane()
```

    ABC
    Zap
    ABC
    


```python
x = 12
if x < 5:
print("smaller")
else:
    print("bigger")
print("all done")
```


      File "<ipython-input-7-adfc3d2d2a4c>", line 3
        print("smaller")
            ^
    IndentationError: expected an indented block
    



```python
zap = "hello there bob"
print(zap[4])
```

    o
    


```python
x
if x < 2 :
    print("Below 2")
elif x < 20 :
    print("Below 20")
elif x < 10 :
    print("Below 10")
else :
    print("Something else")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-2-8c9cb5ba0937> in <module>
    ----> 1 x
          2 if x < 2 :
          3     print("Below 2")
          4 elif x < 20 :
          5     print("Below 20")
    

    NameError: name 'x' is not defined



```python
x=0
```


```python
a = "123"
b = 456
c = a + b
print(c)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-3-833eb22d1061> in <module>
          1 a = "123"
          2 b = 456
    ----> 3 c = a + b
          4 print(c)
    

    TypeError: can only concatenate str (not "int") to str



```python
    line1
try:
    line2
    line3
    line4
except:
     line5
line6
```


      File "<ipython-input-12-8c297cb14973>", line 3
        line2
            ^
    IndentationError: expected an indented block
    



```python
abc = "With three words"
stuff = abc.split()
print(stuff)
```

    ['With', 'three', 'words']
    


```python
abc = "With three words"
stuff = abc.split()
print(len(stuff))
```

    3
    


```python
<person>
   <name>Chuck</name>
   <phone type="intl">
	  +1 734 303 4456
   </phone>
   <email hide="yes" />
</person>
```


      File "<ipython-input-8-77ebd0385f37>", line 1
        <person>
        ^
    SyntaxError: invalid syntax
    



```python

```
