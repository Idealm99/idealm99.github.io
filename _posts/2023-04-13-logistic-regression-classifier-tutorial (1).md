---
layout: single
title:  "jupyter notebook 변환하기!"
categories: coding
tag: [python, blog, jekyll]
toc: true
author_profile: false
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
0
"
>
<
/
a
>


#
 
*
*
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
C
l
a
s
s
i
f
i
e
r
 
T
u
t
o
r
i
a
l
 
w
i
t
h
 
P
y
t
h
o
n
*
*






H
e
l
l
o
 
f
r
i
e
n
d
s
,






I
n
 
t
h
i
s
 
k
e
r
n
e
l
,
 
I
 
i
m
p
l
e
m
e
n
t
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
w
i
t
h
 
P
y
t
h
o
n
 
a
n
d
 
S
c
i
k
i
t
-
L
e
a
r
n
.
 
I
 
b
u
i
l
d
 
a
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
c
l
a
s
s
i
f
i
e
r
 
t
o
 
p
r
e
d
i
c
t
 
w
h
e
t
h
e
r
 
o
r
 
n
o
t
 
i
t
 
w
i
l
l
 
r
a
i
n
 
t
o
m
o
r
r
o
w
 
i
n
 
A
u
s
t
r
a
l
i
a
.
 
I
 
t
r
a
i
n
 
a
 
b
i
n
a
r
y
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
u
s
i
n
g
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
.
 


*
*
A
s
 
a
l
w
a
y
s
,
 
I
 
h
o
p
e
 
y
o
u
 
f
i
n
d
 
t
h
i
s
 
k
e
r
n
e
l
 
u
s
e
f
u
l
 
a
n
d
 
y
o
u
r
 
<
f
o
n
t
 
c
o
l
o
r
=
"
r
e
d
"
>
<
b
>
U
P
V
O
T
E
S
<
/
b
>
<
/
f
o
n
t
>
 
w
o
u
l
d
 
b
e
 
h
i
g
h
l
y
 
a
p
p
r
e
c
i
a
t
e
d
*
*
.




<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
0
.
1
"
>
<
/
a
>


#
 
*
*
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
*
*






1
.
	
[
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
]
(
#
1
)


2
.
	
[
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
i
n
t
u
i
t
i
o
n
]
(
#
2
)


3
.
	
[
A
s
s
u
m
p
t
i
o
n
s
 
o
f
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
]
(
#
3
)


4
.
	
[
T
y
p
e
s
 
o
f
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
]
(
#
4
)


5
.
	
[
I
m
p
o
r
t
 
l
i
b
r
a
r
i
e
s
]
(
#
5
)


6
.
	
[
I
m
p
o
r
t
 
d
a
t
a
s
e
t
]
(
#
6
)


7
.
	
[
E
x
p
l
o
r
a
t
o
r
y
 
d
a
t
a
 
a
n
a
l
y
s
i
s
]
(
#
7
)


8
.
	
[
D
e
c
l
a
r
e
 
f
e
a
t
u
r
e
 
v
e
c
t
o
r
 
a
n
d
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
]
(
#
8
)


9
.
	
[
S
p
l
i
t
 
d
a
t
a
 
i
n
t
o
 
s
e
p
a
r
a
t
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t
]
(
#
9
)


1
0
.
	
[
F
e
a
t
u
r
e
 
e
n
g
i
n
e
e
r
i
n
g
]
(
#
1
0
)


1
1
.
	
[
F
e
a
t
u
r
e
 
s
c
a
l
i
n
g
]
(
#
1
1
)


1
2
.
	
[
M
o
d
e
l
 
t
r
a
i
n
i
n
g
]
(
#
1
2
)


1
3
.
	
[
P
r
e
d
i
c
t
 
r
e
s
u
l
t
s
]
(
#
1
3
)


1
4
.
	
[
C
h
e
c
k
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
]
(
#
1
4
)


1
5
.
	
[
C
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
]
(
#
1
5
)


1
6
.
	
[
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
e
t
r
i
c
e
s
]
(
#
1
6
)


1
7
.
	
[
A
d
j
u
s
t
i
n
g
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
]
(
#
1
7
)


1
8
.
	
[
R
O
C
 
-
 
A
U
C
]
(
#
1
8
)


1
9
.
	
[
k
-
F
o
l
d
 
C
r
o
s
s
 
V
a
l
i
d
a
t
i
o
n
]
(
#
1
9
)


2
0
.
	
[
H
y
p
e
r
p
a
r
a
m
e
t
e
r
 
o
p
t
i
m
i
z
a
t
i
o
n
 
u
s
i
n
g
 
G
r
i
d
S
e
a
r
c
h
 
C
V
]
(
#
2
0
)


2
1
.
	
[
R
e
s
u
l
t
s
 
a
n
d
 
c
o
n
c
l
u
s
i
o
n
]
(
#
2
1
)


2
2
.
 
[
R
e
f
e
r
e
n
c
e
s
]
(
#
2
2
)




#
 
*
*
1
.
 
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






W
h
e
n
 
d
a
t
a
 
s
c
i
e
n
t
i
s
t
s
 
m
a
y
 
c
o
m
e
 
a
c
r
o
s
s
 
a
 
n
e
w
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
p
r
o
b
l
e
m
,
 
t
h
e
 
f
i
r
s
t
 
a
l
g
o
r
i
t
h
m
 
t
h
a
t
 
m
a
y
 
c
o
m
e
 
a
c
r
o
s
s
 
t
h
e
i
r
 
m
i
n
d
 
i
s
 
*
*
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
*
*
.
 
I
t
 
i
s
 
a
 
s
u
p
e
r
v
i
s
e
d
 
l
e
a
r
n
i
n
g
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
l
g
o
r
i
t
h
m
 
w
h
i
c
h
 
i
s
 
u
s
e
d
 
t
o
 
p
r
e
d
i
c
t
 
o
b
s
e
r
v
a
t
i
o
n
s
 
t
o
 
a
 
d
i
s
c
r
e
t
e
 
s
e
t
 
o
f
 
c
l
a
s
s
e
s
.
 
P
r
a
c
t
i
c
a
l
l
y
,
 
i
t
 
i
s
 
u
s
e
d
 
t
o
 
c
l
a
s
s
i
f
y
 
o
b
s
e
r
v
a
t
i
o
n
s
 
i
n
t
o
 
d
i
f
f
e
r
e
n
t
 
c
a
t
e
g
o
r
i
e
s
.
 
H
e
n
c
e
,
 
i
t
s
 
o
u
t
p
u
t
 
i
s
 
d
i
s
c
r
e
t
e
 
i
n
 
n
a
t
u
r
e
.
 
*
*
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
*
*
 
i
s
 
a
l
s
o
 
c
a
l
l
e
d
 
*
*
L
o
g
i
t
 
R
e
g
r
e
s
s
i
o
n
*
*
.
 
I
t
 
i
s
 
o
n
e
 
o
f
 
t
h
e
 
m
o
s
t
 
s
i
m
p
l
e
,
 
s
t
r
a
i
g
h
t
f
o
r
w
a
r
d
 
a
n
d
 
v
e
r
s
a
t
i
l
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
l
g
o
r
i
t
h
m
s
 
w
h
i
c
h
 
i
s
 
u
s
e
d
 
t
o
 
s
o
l
v
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
p
r
o
b
l
e
m
s
.


#
 
*
*
2
.
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
i
n
t
u
i
t
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
2
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






I
n
 
s
t
a
t
i
s
t
i
c
s
,
 
t
h
e
 
*
*
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
*
*
 
i
s
 
a
 
w
i
d
e
l
y
 
u
s
e
d
 
s
t
a
t
i
s
t
i
c
a
l
 
m
o
d
e
l
 
w
h
i
c
h
 
i
s
 
p
r
i
m
a
r
i
l
y
 
u
s
e
d
 
f
o
r
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
p
u
r
p
o
s
e
s
.
 
I
t
 
m
e
a
n
s
 
t
h
a
t
 
g
i
v
e
n
 
a
 
s
e
t
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
,
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
a
l
g
o
r
i
t
h
m
 
h
e
l
p
s
 
u
s
 
t
o
 
c
l
a
s
s
i
f
y
 
t
h
e
s
e
 
o
b
s
e
r
v
a
t
i
o
n
s
 
i
n
t
o
 
t
w
o
 
o
r
 
m
o
r
e
 
d
i
s
c
r
e
t
e
 
c
l
a
s
s
e
s
.
 
S
o
,
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
i
s
 
d
i
s
c
r
e
t
e
 
i
n
 
n
a
t
u
r
e
.






T
h
e
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
a
l
g
o
r
i
t
h
m
 
w
o
r
k
s
 
a
s
 
f
o
l
l
o
w
s
 
-


#
#
 
*
*
I
m
p
l
e
m
e
n
t
 
l
i
n
e
a
r
 
e
q
u
a
t
i
o
n
*
*






L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
a
l
g
o
r
i
t
h
m
 
w
o
r
k
s
 
b
y
 
i
m
p
l
e
m
e
n
t
i
n
g
 
a
 
l
i
n
e
a
r
 
e
q
u
a
t
i
o
n
 
w
i
t
h
 
i
n
d
e
p
e
n
d
e
n
t
 
o
r
 
e
x
p
l
a
n
a
t
o
r
y
 
v
a
r
i
a
b
l
e
s
 
t
o
 
p
r
e
d
i
c
t
 
a
 
r
e
s
p
o
n
s
e
 
v
a
l
u
e
.
 
F
o
r
 
e
x
a
m
p
l
e
,
 
w
e
 
c
o
n
s
i
d
e
r
 
t
h
e
 
e
x
a
m
p
l
e
 
o
f
 
n
u
m
b
e
r
 
o
f
 
h
o
u
r
s
 
s
t
u
d
i
e
d
 
a
n
d
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
p
a
s
s
i
n
g
 
t
h
e
 
e
x
a
m
.
 
H
e
r
e
,
 
n
u
m
b
e
r
 
o
f
 
h
o
u
r
s
 
s
t
u
d
i
e
d
 
i
s
 
t
h
e
 
e
x
p
l
a
n
a
t
o
r
y
 
v
a
r
i
a
b
l
e
 
a
n
d
 
i
t
 
i
s
 
d
e
n
o
t
e
d
 
b
y
 
x
1
.
 
P
r
o
b
a
b
i
l
i
t
y
 
o
f
 
p
a
s
s
i
n
g
 
t
h
e
 
e
x
a
m
 
i
s
 
t
h
e
 
r
e
s
p
o
n
s
e
 
o
r
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
a
n
d
 
i
t
 
i
s
 
d
e
n
o
t
e
d
 
b
y
 
z
.






I
f
 
w
e
 
h
a
v
e
 
o
n
e
 
e
x
p
l
a
n
a
t
o
r
y
 
v
a
r
i
a
b
l
e
 
(
x
1
)
 
a
n
d
 
o
n
e
 
r
e
s
p
o
n
s
e
 
v
a
r
i
a
b
l
e
 
(
z
)
,
 
t
h
e
n
 
t
h
e
 
l
i
n
e
a
r
 
e
q
u
a
t
i
o
n
 
w
o
u
l
d
 
b
e
 
g
i
v
e
n
 
m
a
t
h
e
m
a
t
i
c
a
l
l
y
 
w
i
t
h
 
t
h
e
 
f
o
l
l
o
w
i
n
g
 
e
q
u
a
t
i
o
n
-




 
 
 
 
z
 
=
 
β
0
 
+
 
β
1
x
1
 
 
 
 




H
e
r
e
,
 
t
h
e
 
c
o
e
f
f
i
c
i
e
n
t
s
 
β
0
 
a
n
d
 
β
1
 
a
r
e
 
t
h
e
 
p
a
r
a
m
e
t
e
r
s
 
o
f
 
t
h
e
 
m
o
d
e
l
.






I
f
 
t
h
e
r
e
 
a
r
e
 
m
u
l
t
i
p
l
e
 
e
x
p
l
a
n
a
t
o
r
y
 
v
a
r
i
a
b
l
e
s
,
 
t
h
e
n
 
t
h
e
 
a
b
o
v
e
 
e
q
u
a
t
i
o
n
 
c
a
n
 
b
e
 
e
x
t
e
n
d
e
d
 
t
o




 
 
 
 
z
 
=
 
β
0
 
+
 
β
1
x
1
+
 
β
2
x
2
+
…
…
.
.
+
 
β
n
x
n


 
 
 
 


H
e
r
e
,
 
t
h
e
 
c
o
e
f
f
i
c
i
e
n
t
s
 
β
0
,
 
β
1
,
 
β
2
 
a
n
d
 
β
n
 
a
r
e
 
t
h
e
 
p
a
r
a
m
e
t
e
r
s
 
o
f
 
t
h
e
 
m
o
d
e
l
.




S
o
,
 
t
h
e
 
p
r
e
d
i
c
t
e
d
 
r
e
s
p
o
n
s
e
 
v
a
l
u
e
 
i
s
 
g
i
v
e
n
 
b
y
 
t
h
e
 
a
b
o
v
e
 
e
q
u
a
t
i
o
n
s
 
a
n
d
 
i
s
 
d
e
n
o
t
e
d
 
b
y
 
z
.


#
#
 
*
*
S
i
g
m
o
i
d
 
F
u
n
c
t
i
o
n
*
*




T
h
i
s
 
p
r
e
d
i
c
t
e
d
 
r
e
s
p
o
n
s
e
 
v
a
l
u
e
,
 
d
e
n
o
t
e
d
 
b
y
 
z
 
i
s
 
t
h
e
n
 
c
o
n
v
e
r
t
e
d
 
i
n
t
o
 
a
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
t
h
a
t
 
l
i
e
 
b
e
t
w
e
e
n
 
0
 
a
n
d
 
1
.
 
W
e
 
u
s
e
 
t
h
e
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
i
n
 
o
r
d
e
r
 
t
o
 
m
a
p
 
p
r
e
d
i
c
t
e
d
 
v
a
l
u
e
s
 
t
o
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
s
.
 
T
h
i
s
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
t
h
e
n
 
m
a
p
s
 
a
n
y
 
r
e
a
l
 
v
a
l
u
e
 
i
n
t
o
 
a
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
b
e
t
w
e
e
n
 
0
 
a
n
d
 
1
.




I
n
 
m
a
c
h
i
n
e
 
l
e
a
r
n
i
n
g
,
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
i
s
 
u
s
e
d
 
t
o
 
m
a
p
 
p
r
e
d
i
c
t
i
o
n
s
 
t
o
 
p
r
o
b
a
b
i
l
i
t
i
e
s
.
 
T
h
e
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
h
a
s
 
a
n
 
S
 
s
h
a
p
e
d
 
c
u
r
v
e
.
 
I
t
 
i
s
 
a
l
s
o
 
c
a
l
l
e
d
 
s
i
g
m
o
i
d
 
c
u
r
v
e
.




A
 
S
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
i
s
 
a
 
s
p
e
c
i
a
l
 
c
a
s
e
 
o
f
 
t
h
e
 
L
o
g
i
s
t
i
c
 
f
u
n
c
t
i
o
n
.
 
I
t
 
i
s
 
g
i
v
e
n
 
b
y
 
t
h
e
 
f
o
l
l
o
w
i
n
g
 
m
a
t
h
e
m
a
t
i
c
a
l
 
f
o
r
m
u
l
a
.




G
r
a
p
h
i
c
a
l
l
y
,
 
w
e
 
c
a
n
 
r
e
p
r
e
s
e
n
t
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
w
i
t
h
 
t
h
e
 
f
o
l
l
o
w
i
n
g
 
g
r
a
p
h
.


#
#
#
 
S
i
g
m
o
i
d
 
F
u
n
c
t
i
o
n




!
[
S
i
g
m
o
i
d
 
F
u
n
c
t
i
o
n
]
(
h
t
t
p
s
:
/
/
m
i
r
o
.
m
e
d
i
u
m
.
c
o
m
/
m
a
x
/
9
7
0
/
1
*
X
u
7
B
5
y
9
g
p
0
i
L
5
o
o
B
j
7
L
t
W
w
.
p
n
g
)


#
#
 
*
*
D
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
*
*




T
h
e
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
r
e
t
u
r
n
s
 
a
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
b
e
t
w
e
e
n
 
0
 
a
n
d
 
1
.
 
T
h
i
s
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
i
s
 
t
h
e
n
 
m
a
p
p
e
d
 
t
o
 
a
 
d
i
s
c
r
e
t
e
 
c
l
a
s
s
 
w
h
i
c
h
 
i
s
 
e
i
t
h
e
r
 
“
0
”
 
o
r
 
“
1
”
.
 
I
n
 
o
r
d
e
r
 
t
o
 
m
a
p
 
t
h
i
s
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
t
o
 
a
 
d
i
s
c
r
e
t
e
 
c
l
a
s
s
 
(
p
a
s
s
/
f
a
i
l
,
 
y
e
s
/
n
o
,
 
t
r
u
e
/
f
a
l
s
e
)
,
 
w
e
 
s
e
l
e
c
t
 
a
 
t
h
r
e
s
h
o
l
d
 
v
a
l
u
e
.
 
T
h
i
s
 
t
h
r
e
s
h
o
l
d
 
v
a
l
u
e
 
i
s
 
c
a
l
l
e
d
 
D
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
.
 
A
b
o
v
e
 
t
h
i
s
 
t
h
r
e
s
h
o
l
d
 
v
a
l
u
e
,
 
w
e
 
w
i
l
l
 
m
a
p
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
s
 
i
n
t
o
 
c
l
a
s
s
 
1
 
a
n
d
 
b
e
l
o
w
 
w
h
i
c
h
 
w
e
 
w
i
l
l
 
m
a
p
 
v
a
l
u
e
s
 
i
n
t
o
 
c
l
a
s
s
 
0
.




M
a
t
h
e
m
a
t
i
c
a
l
l
y
,
 
i
t
 
c
a
n
 
b
e
 
e
x
p
r
e
s
s
e
d
 
a
s
 
f
o
l
l
o
w
s
:
-




p
 
≥
 
0
.
5
 
=
>
 
c
l
a
s
s
 
=
 
1




p
 
<
 
0
.
5
 
=
>
 
c
l
a
s
s
 
=
 
0
 




G
e
n
e
r
a
l
l
y
,
 
t
h
e
 
d
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
 
i
s
 
s
e
t
 
t
o
 
0
.
5
.
 
S
o
,
 
i
f
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
i
s
 
0
.
8
 
(
>
 
0
.
5
)
,
 
w
e
 
w
i
l
l
 
m
a
p
 
t
h
i
s
 
o
b
s
e
r
v
a
t
i
o
n
 
t
o
 
c
l
a
s
s
 
1
.
 
S
i
m
i
l
a
r
l
y
,
 
i
f
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
v
a
l
u
e
 
i
s
 
0
.
2
 
(
<
 
0
.
5
)
,
 
w
e
 
w
i
l
l
 
m
a
p
 
t
h
i
s
 
o
b
s
e
r
v
a
t
i
o
n
 
t
o
 
c
l
a
s
s
 
0
.
 
T
h
i
s
 
i
s
 
r
e
p
r
e
s
e
n
t
e
d
 
i
n
 
t
h
e
 
g
r
a
p
h
 
b
e
l
o
w
-


!
[
D
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
 
i
n
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
]
(
h
t
t
p
s
:
/
/
m
l
-
c
h
e
a
t
s
h
e
e
t
.
r
e
a
d
t
h
e
d
o
c
s
.
i
o
/
e
n
/
l
a
t
e
s
t
/
_
i
m
a
g
e
s
/
l
o
g
i
s
t
i
c
_
r
e
g
r
e
s
s
i
o
n
_
s
i
g
m
o
i
d
_
w
_
t
h
r
e
s
h
o
l
d
.
p
n
g
)


#
#
 
*
*
M
a
k
i
n
g
 
p
r
e
d
i
c
t
i
o
n
s
*
*




N
o
w
,
 
w
e
 
k
n
o
w
 
a
b
o
u
t
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
a
n
d
 
d
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
 
i
n
 
l
o
g
i
s
t
i
c
 
r
e
g
r
e
s
s
i
o
n
.
 
W
e
 
c
a
n
 
u
s
e
 
o
u
r
 
k
n
o
w
l
e
d
g
e
 
o
f
 
s
i
g
m
o
i
d
 
f
u
n
c
t
i
o
n
 
a
n
d
 
d
e
c
i
s
i
o
n
 
b
o
u
n
d
a
r
y
 
t
o
 
w
r
i
t
e
 
a
 
p
r
e
d
i
c
t
i
o
n
 
f
u
n
c
t
i
o
n
.
 
A
 
p
r
e
d
i
c
t
i
o
n
 
f
u
n
c
t
i
o
n
 
i
n
 
l
o
g
i
s
t
i
c
 
r
e
g
r
e
s
s
i
o
n
 
r
e
t
u
r
n
s
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
b
e
i
n
g
 
p
o
s
i
t
i
v
e
,
 
Y
e
s
 
o
r
 
T
r
u
e
.
 
W
e
 
c
a
l
l
 
t
h
i
s
 
a
s
 
c
l
a
s
s
 
1
 
a
n
d
 
i
t
 
i
s
 
d
e
n
o
t
e
d
 
b
y
 
P
(
c
l
a
s
s
 
=
 
1
)
.
 
I
f
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
y
 
i
n
c
h
e
s
 
c
l
o
s
e
r
 
t
o
 
o
n
e
,
 
t
h
e
n
 
w
e
 
w
i
l
l
 
b
e
 
m
o
r
e
 
c
o
n
f
i
d
e
n
t
 
a
b
o
u
t
 
o
u
r
 
m
o
d
e
l
 
t
h
a
t
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
i
s
 
i
n
 
c
l
a
s
s
 
1
,
 
o
t
h
e
r
w
i
s
e
 
i
t
 
i
s
 
i
n
 
c
l
a
s
s
 
0
.




#
 
*
*
3
.
 
A
s
s
u
m
p
t
i
o
n
s
 
o
f
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
3
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






T
h
e
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
r
e
q
u
i
r
e
s
 
s
e
v
e
r
a
l
 
k
e
y
 
a
s
s
u
m
p
t
i
o
n
s
.
 
T
h
e
s
e
 
a
r
e
 
a
s
 
f
o
l
l
o
w
s
:
-




1
.
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
r
e
q
u
i
r
e
s
 
t
h
e
 
d
e
p
e
n
d
e
n
t
 
v
a
r
i
a
b
l
e
 
t
o
 
b
e
 
b
i
n
a
r
y
,
 
m
u
l
t
i
n
o
m
i
a
l
 
o
r
 
o
r
d
i
n
a
l
 
i
n
 
n
a
t
u
r
e
.




2
.
 
I
t
 
r
e
q
u
i
r
e
s
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
s
 
t
o
 
b
e
 
i
n
d
e
p
e
n
d
e
n
t
 
o
f
 
e
a
c
h
 
o
t
h
e
r
.
 
S
o
,
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
s
 
s
h
o
u
l
d
 
n
o
t
 
c
o
m
e
 
f
r
o
m
 
r
e
p
e
a
t
e
d
 
m
e
a
s
u
r
e
m
e
n
t
s
.




3
.
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
a
l
g
o
r
i
t
h
m
 
r
e
q
u
i
r
e
s
 
l
i
t
t
l
e
 
o
r
 
n
o
 
m
u
l
t
i
c
o
l
l
i
n
e
a
r
i
t
y
 
a
m
o
n
g
 
t
h
e
 
i
n
d
e
p
e
n
d
e
n
t
 
v
a
r
i
a
b
l
e
s
.
 
I
t
 
m
e
a
n
s
 
t
h
a
t
 
t
h
e
 
i
n
d
e
p
e
n
d
e
n
t
 
v
a
r
i
a
b
l
e
s
 
s
h
o
u
l
d
 
n
o
t
 
b
e
 
t
o
o
 
h
i
g
h
l
y
 
c
o
r
r
e
l
a
t
e
d
 
w
i
t
h
 
e
a
c
h
 
o
t
h
e
r
.




4
.
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
a
s
s
u
m
e
s
 
l
i
n
e
a
r
i
t
y
 
o
f
 
i
n
d
e
p
e
n
d
e
n
t
 
v
a
r
i
a
b
l
e
s
 
a
n
d
 
l
o
g
 
o
d
d
s
.




5
.
 
T
h
e
 
s
u
c
c
e
s
s
 
o
f
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
d
e
p
e
n
d
s
 
o
n
 
t
h
e
 
s
a
m
p
l
e
 
s
i
z
e
s
.
 
T
y
p
i
c
a
l
l
y
,
 
i
t
 
r
e
q
u
i
r
e
s
 
a
 
l
a
r
g
e
 
s
a
m
p
l
e
 
s
i
z
e
 
t
o
 
a
c
h
i
e
v
e
 
t
h
e
 
h
i
g
h
 
a
c
c
u
r
a
c
y
.


#
 
*
*
4
.
 
T
y
p
e
s
 
o
f
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
4
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
c
a
n
 
b
e
 
c
l
a
s
s
i
f
i
e
d
 
i
n
t
o
 
t
h
r
e
e
 
g
r
o
u
p
s
 
b
a
s
e
d
 
o
n
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
c
a
t
e
g
o
r
i
e
s
.
 
T
h
e
s
e
 
t
h
r
e
e
 
g
r
o
u
p
s
 
a
r
e
 
d
e
s
c
r
i
b
e
d
 
b
e
l
o
w
:
-




#
#
#
 
1
.
 
B
i
n
a
r
y
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n




I
n
 
B
i
n
a
r
y
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
,
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
h
a
s
 
t
w
o
 
p
o
s
s
i
b
l
e
 
c
a
t
e
g
o
r
i
e
s
.
 
T
h
e
 
c
o
m
m
o
n
 
e
x
a
m
p
l
e
s
 
o
f
 
c
a
t
e
g
o
r
i
e
s
 
a
r
e
 
y
e
s
 
o
r
 
n
o
,
 
g
o
o
d
 
o
r
 
b
a
d
,
 
t
r
u
e
 
o
r
 
f
a
l
s
e
,
 
s
p
a
m
 
o
r
 
n
o
 
s
p
a
m
 
a
n
d
 
p
a
s
s
 
o
r
 
f
a
i
l
.






#
#
#
 
2
.
 
M
u
l
t
i
n
o
m
i
a
l
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n




I
n
 
M
u
l
t
i
n
o
m
i
a
l
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
,
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
h
a
s
 
t
h
r
e
e
 
o
r
 
m
o
r
e
 
c
a
t
e
g
o
r
i
e
s
 
w
h
i
c
h
 
a
r
e
 
n
o
t
 
i
n
 
a
n
y
 
p
a
r
t
i
c
u
l
a
r
 
o
r
d
e
r
.
 
S
o
,
 
t
h
e
r
e
 
a
r
e
 
t
h
r
e
e
 
o
r
 
m
o
r
e
 
n
o
m
i
n
a
l
 
c
a
t
e
g
o
r
i
e
s
.
 
T
h
e
 
e
x
a
m
p
l
e
s
 
i
n
c
l
u
d
e
 
t
h
e
 
t
y
p
e
 
o
f
 
c
a
t
e
g
o
r
i
e
s
 
o
f
 
f
r
u
i
t
s
 
-
 
a
p
p
l
e
,
 
m
a
n
g
o
,
 
o
r
a
n
g
e
 
a
n
d
 
b
a
n
a
n
a
.






#
#
#
 
3
.
 
O
r
d
i
n
a
l
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n




I
n
 
O
r
d
i
n
a
l
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
,
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
 
h
a
s
 
t
h
r
e
e
 
o
r
 
m
o
r
e
 
o
r
d
i
n
a
l
 
c
a
t
e
g
o
r
i
e
s
.
 
S
o
,
 
t
h
e
r
e
 
i
s
 
i
n
t
r
i
n
s
i
c
 
o
r
d
e
r
 
i
n
v
o
l
v
e
d
 
w
i
t
h
 
t
h
e
 
c
a
t
e
g
o
r
i
e
s
.
 
F
o
r
 
e
x
a
m
p
l
e
,
 
t
h
e
 
s
t
u
d
e
n
t
 
p
e
r
f
o
r
m
a
n
c
e
 
c
a
n
 
b
e
 
c
a
t
e
g
o
r
i
z
e
d
 
a
s
 
p
o
o
r
,
 
a
v
e
r
a
g
e
,
 
g
o
o
d
 
a
n
d
 
e
x
c
e
l
l
e
n
t
.




#
 
*
*
5
.
 
I
m
p
o
r
t
 
l
i
b
r
a
r
i
e
s
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
5
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
#
 
T
h
i
s
 
P
y
t
h
o
n
 
3
 
e
n
v
i
r
o
n
m
e
n
t
 
c
o
m
e
s
 
w
i
t
h
 
m
a
n
y
 
h
e
l
p
f
u
l
 
a
n
a
l
y
t
i
c
s
 
l
i
b
r
a
r
i
e
s
 
i
n
s
t
a
l
l
e
d

#
 
I
t
 
i
s
 
d
e
f
i
n
e
d
 
b
y
 
t
h
e
 
k
a
g
g
l
e
/
p
y
t
h
o
n
 
d
o
c
k
e
r
 
i
m
a
g
e
:
 
h
t
t
p
s
:
/
/
g
i
t
h
u
b
.
c
o
m
/
k
a
g
g
l
e
/
d
o
c
k
e
r
-
p
y
t
h
o
n

#
 
F
o
r
 
e
x
a
m
p
l
e
,
 
h
e
r
e
'
s
 
s
e
v
e
r
a
l
 
h
e
l
p
f
u
l
 
p
a
c
k
a
g
e
s
 
t
o
 
l
o
a
d
 
i
n
 


i
m
p
o
r
t
 
n
u
m
p
y
 
a
s
 
n
p
 
#
 
l
i
n
e
a
r
 
a
l
g
e
b
r
a

i
m
p
o
r
t
 
p
a
n
d
a
s
 
a
s
 
p
d
 
#
 
d
a
t
a
 
p
r
o
c
e
s
s
i
n
g
,
 
C
S
V
 
f
i
l
e
 
I
/
O
 
(
e
.
g
.
 
p
d
.
r
e
a
d
_
c
s
v
)

i
m
p
o
r
t
 
m
a
t
p
l
o
t
l
i
b
.
p
y
p
l
o
t
 
a
s
 
p
l
t
 
#
 
d
a
t
a
 
v
i
s
u
a
l
i
z
a
t
i
o
n

i
m
p
o
r
t
 
s
e
a
b
o
r
n
 
a
s
 
s
n
s
 
#
 
s
t
a
t
i
s
t
i
c
a
l
 
d
a
t
a
 
v
i
s
u
a
l
i
z
a
t
i
o
n

%
m
a
t
p
l
o
t
l
i
b
 
i
n
l
i
n
e


#
 
I
n
p
u
t
 
d
a
t
a
 
f
i
l
e
s
 
a
r
e
 
a
v
a
i
l
a
b
l
e
 
i
n
 
t
h
e
 
"
.
.
/
i
n
p
u
t
/
"
 
d
i
r
e
c
t
o
r
y
.

#
 
F
o
r
 
e
x
a
m
p
l
e
,
 
r
u
n
n
i
n
g
 
t
h
i
s
 
(
b
y
 
c
l
i
c
k
i
n
g
 
r
u
n
 
o
r
 
p
r
e
s
s
i
n
g
 
S
h
i
f
t
+
E
n
t
e
r
)
 
w
i
l
l
 
l
i
s
t
 
a
l
l
 
f
i
l
e
s
 
u
n
d
e
r
 
t
h
e
 
i
n
p
u
t
 
d
i
r
e
c
t
o
r
y


i
m
p
o
r
t
 
o
s

f
o
r
 
d
i
r
n
a
m
e
,
 
_
,
 
f
i
l
e
n
a
m
e
s
 
i
n
 
o
s
.
w
a
l
k
(
'
/
k
a
g
g
l
e
/
i
n
p
u
t
'
)
:

 
 
 
 
f
o
r
 
f
i
l
e
n
a
m
e
 
i
n
 
f
i
l
e
n
a
m
e
s
:

 
 
 
 
 
 
 
 
p
r
i
n
t
(
o
s
.
p
a
t
h
.
j
o
i
n
(
d
i
r
n
a
m
e
,
 
f
i
l
e
n
a
m
e
)
)


#
 
A
n
y
 
r
e
s
u
l
t
s
 
y
o
u
 
w
r
i
t
e
 
t
o
 
t
h
e
 
c
u
r
r
e
n
t
 
d
i
r
e
c
t
o
r
y
 
a
r
e
 
s
a
v
e
d
 
a
s
 
o
u
t
p
u
t
.

```


```python
i
m
p
o
r
t
 
w
a
r
n
i
n
g
s


w
a
r
n
i
n
g
s
.
f
i
l
t
e
r
w
a
r
n
i
n
g
s
(
'
i
g
n
o
r
e
'
)
```

#
 
*
*
6
.
 
I
m
p
o
r
t
 
d
a
t
a
s
e
t
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
6
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
d
a
t
a
 
=
 
'
/
k
a
g
g
l
e
/
i
n
p
u
t
/
w
e
a
t
h
e
r
-
d
a
t
a
s
e
t
-
r
a
t
t
l
e
-
p
a
c
k
a
g
e
/
w
e
a
t
h
e
r
A
U
S
.
c
s
v
'


d
f
 
=
 
p
d
.
r
e
a
d
_
c
s
v
(
d
a
t
a
)
```

#
 
*
*
7
.
 
E
x
p
l
o
r
a
t
o
r
y
 
d
a
t
a
 
a
n
a
l
y
s
i
s
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
7
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






N
o
w
,
 
I
 
w
i
l
l
 
e
x
p
l
o
r
e
 
t
h
e
 
d
a
t
a
 
t
o
 
g
a
i
n
 
i
n
s
i
g
h
t
s
 
a
b
o
u
t
 
t
h
e
 
d
a
t
a
.
 



```python
#
 
v
i
e
w
 
d
i
m
e
n
s
i
o
n
s
 
o
f
 
d
a
t
a
s
e
t


d
f
.
s
h
a
p
e
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
1
4
2
1
9
3
 
i
n
s
t
a
n
c
e
s
 
a
n
d
 
2
4
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
h
e
 
d
a
t
a
 
s
e
t
.



```python
#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t


d
f
.
h
e
a
d
(
)
```


```python
c
o
l
_
n
a
m
e
s
 
=
 
d
f
.
c
o
l
u
m
n
s


c
o
l
_
n
a
m
e
s
```

#
#
#
 
D
r
o
p
 
 
R
I
S
K
_
M
M
 
v
a
r
i
a
b
l
e




I
t
 
i
s
 
g
i
v
e
n
 
i
n
 
t
h
e
 
d
a
t
a
s
e
t
 
d
e
s
c
r
i
p
t
i
o
n
,
 
t
h
a
t
 
w
e
 
s
h
o
u
l
d
 
d
r
o
p
 
t
h
e
 
`
R
I
S
K
_
M
M
`
 
f
e
a
t
u
r
e
 
v
a
r
i
a
b
l
e
 
f
r
o
m
 
t
h
e
 
d
a
t
a
s
e
t
 
d
e
s
c
r
i
p
t
i
o
n
.
 
S
o
,
 
w
e
 


s
h
o
u
l
d
 
d
r
o
p
 
i
t
 
a
s
 
f
o
l
l
o
w
s
-



```python
d
f
.
d
r
o
p
(
[
'
R
I
S
K
_
M
M
'
]
,
 
a
x
i
s
=
1
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)
```


```python
#
 
v
i
e
w
 
s
u
m
m
a
r
y
 
o
f
 
d
a
t
a
s
e
t


d
f
.
i
n
f
o
(
)
```

#
#
#
 
T
y
p
e
s
 
o
f
 
v
a
r
i
a
b
l
e
s






I
n
 
t
h
i
s
 
s
e
c
t
i
o
n
,
 
I
 
s
e
g
r
e
g
a
t
e
 
t
h
e
 
d
a
t
a
s
e
t
 
i
n
t
o
 
c
a
t
e
g
o
r
i
c
a
l
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.
 
T
h
e
r
e
 
a
r
e
 
a
 
m
i
x
t
u
r
e
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
h
e
 
d
a
t
a
s
e
t
.
 
C
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
h
a
v
e
 
d
a
t
a
 
t
y
p
e
 
o
b
j
e
c
t
.
 
N
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
h
a
v
e
 
d
a
t
a
 
t
y
p
e
 
f
l
o
a
t
6
4
.






F
i
r
s
t
 
o
f
 
a
l
l
,
 
I
 
w
i
l
l
 
f
i
n
d
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.



```python
#
 
f
i
n
d
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


c
a
t
e
g
o
r
i
c
a
l
 
=
 
[
v
a
r
 
f
o
r
 
v
a
r
 
i
n
 
d
f
.
c
o
l
u
m
n
s
 
i
f
 
d
f
[
v
a
r
]
.
d
t
y
p
e
=
=
'
O
'
]


p
r
i
n
t
(
'
T
h
e
r
e
 
a
r
e
 
{
}
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
\
n
'
.
f
o
r
m
a
t
(
l
e
n
(
c
a
t
e
g
o
r
i
c
a
l
)
)
)


p
r
i
n
t
(
'
T
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
a
r
e
 
:
'
,
 
c
a
t
e
g
o
r
i
c
a
l
)
```


```python
#
 
v
i
e
w
 
t
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


d
f
[
c
a
t
e
g
o
r
i
c
a
l
]
.
h
e
a
d
(
)
```

#
#
#
 
S
u
m
m
a
r
y
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






-
 
T
h
e
r
e
 
i
s
 
a
 
d
a
t
e
 
v
a
r
i
a
b
l
e
.
 
I
t
 
i
s
 
d
e
n
o
t
e
d
 
b
y
 
`
D
a
t
e
`
 
c
o
l
u
m
n
.






-
 
T
h
e
r
e
 
a
r
e
 
6
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.
 
T
h
e
s
e
 
a
r
e
 
g
i
v
e
n
 
b
y
 
`
L
o
c
a
t
i
o
n
`
,
 
`
W
i
n
d
G
u
s
t
D
i
r
`
,
 
`
W
i
n
d
D
i
r
9
a
m
`
,
 
`
W
i
n
d
D
i
r
3
p
m
`
,
 
`
R
a
i
n
T
o
d
a
y
`
 
a
n
d
 
 
`
R
a
i
n
T
o
m
o
r
r
o
w
`
.






-
 
T
h
e
r
e
 
a
r
e
 
t
w
o
 
b
i
n
a
r
y
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
-
 
`
R
a
i
n
T
o
d
a
y
`
 
a
n
d
 
 
`
R
a
i
n
T
o
m
o
r
r
o
w
`
.






-
 
`
R
a
i
n
T
o
m
o
r
r
o
w
`
 
i
s
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
.


#
#
 
E
x
p
l
o
r
e
 
p
r
o
b
l
e
m
s
 
w
i
t
h
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






F
i
r
s
t
,
 
I
 
w
i
l
l
 
e
x
p
l
o
r
e
 
t
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.






#
#
#
 
M
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s



```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


d
f
[
c
a
t
e
g
o
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
p
r
i
n
t
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
c
o
n
t
a
i
n
i
n
g
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s


c
a
t
1
 
=
 
[
v
a
r
 
f
o
r
 
v
a
r
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
i
f
 
d
f
[
v
a
r
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
!
=
0
]


p
r
i
n
t
(
d
f
[
c
a
t
1
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
o
n
l
y
 
4
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
h
e
 
d
a
t
a
s
e
t
 
w
h
i
c
h
 
c
o
n
t
a
i
n
s
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
.
 
T
h
e
s
e
 
a
r
e
 
`
W
i
n
d
G
u
s
t
D
i
r
`
,
 
`
W
i
n
d
D
i
r
9
a
m
`
,
 
`
W
i
n
d
D
i
r
3
p
m
`
 
a
n
d
 
`
R
a
i
n
T
o
d
a
y
`
.


#
#
#
 
F
r
e
q
u
e
n
c
y
 
c
o
u
n
t
s
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






N
o
w
,
 
I
 
w
i
l
l
 
c
h
e
c
k
 
t
h
e
 
f
r
e
q
u
e
n
c
y
 
c
o
u
n
t
s
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.



```python
#
 
v
i
e
w
 
f
r
e
q
u
e
n
c
y
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


f
o
r
 
v
a
r
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
:
 

 
 
 
 

 
 
 
 
p
r
i
n
t
(
d
f
[
v
a
r
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
)
```


```python
#
 
v
i
e
w
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


f
o
r
 
v
a
r
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
:
 

 
 
 
 

 
 
 
 
p
r
i
n
t
(
d
f
[
v
a
r
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
/
n
p
.
f
l
o
a
t
(
l
e
n
(
d
f
)
)
)
```

#
#
#
 
N
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
:
 
c
a
r
d
i
n
a
l
i
t
y






T
h
e
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
w
i
t
h
i
n
 
a
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
 
i
s
 
k
n
o
w
n
 
a
s
 
*
*
c
a
r
d
i
n
a
l
i
t
y
*
*
.
 
A
 
h
i
g
h
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
w
i
t
h
i
n
 
a
 
v
a
r
i
a
b
l
e
 
i
s
 
k
n
o
w
n
 
a
s
 
*
*
h
i
g
h
 
c
a
r
d
i
n
a
l
i
t
y
*
*
.
 
H
i
g
h
 
c
a
r
d
i
n
a
l
i
t
y
 
m
a
y
 
p
o
s
e
 
s
o
m
e
 
s
e
r
i
o
u
s
 
p
r
o
b
l
e
m
s
 
i
n
 
t
h
e
 
m
a
c
h
i
n
e
 
l
e
a
r
n
i
n
g
 
m
o
d
e
l
.
 
S
o
,
 
I
 
w
i
l
l
 
c
h
e
c
k
 
f
o
r
 
h
i
g
h
 
c
a
r
d
i
n
a
l
i
t
y
.



```python
#
 
c
h
e
c
k
 
f
o
r
 
c
a
r
d
i
n
a
l
i
t
y
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


f
o
r
 
v
a
r
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
:

 
 
 
 

 
 
 
 
p
r
i
n
t
(
v
a
r
,
 
'
 
c
o
n
t
a
i
n
s
 
'
,
 
l
e
n
(
d
f
[
v
a
r
]
.
u
n
i
q
u
e
(
)
)
,
 
'
 
l
a
b
e
l
s
'
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
i
s
 
a
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
 
w
h
i
c
h
 
n
e
e
d
s
 
t
o
 
b
e
 
p
r
e
p
r
o
c
e
s
s
e
d
.
 
I
 
w
i
l
l
 
d
o
 
p
r
e
p
r
o
c
e
s
s
i
n
g
 
i
n
 
t
h
e
 
f
o
l
l
o
w
i
n
g
 
s
e
c
t
i
o
n
.






A
l
l
 
t
h
e
 
o
t
h
e
r
 
v
a
r
i
a
b
l
e
s
 
c
o
n
t
a
i
n
 
r
e
l
a
t
i
v
e
l
y
 
s
m
a
l
l
e
r
 
n
u
m
b
e
r
 
o
f
 
v
a
r
i
a
b
l
e
s
.


#
#
#
 
F
e
a
t
u
r
e
 
E
n
g
i
n
e
e
r
i
n
g
 
o
f
 
D
a
t
e
 
V
a
r
i
a
b
l
e



```python
d
f
[
'
D
a
t
e
'
]
.
d
t
y
p
e
s
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
 
d
a
t
a
 
t
y
p
e
 
o
f
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
 
i
s
 
o
b
j
e
c
t
.
 
I
 
w
i
l
l
 
p
a
r
s
e
 
t
h
e
 
d
a
t
e
 
c
u
r
r
e
n
t
l
y
 
c
o
d
e
d
 
a
s
 
o
b
j
e
c
t
 
i
n
t
o
 
d
a
t
e
t
i
m
e
 
f
o
r
m
a
t
.



```python
#
 
p
a
r
s
e
 
t
h
e
 
d
a
t
e
s
,
 
c
u
r
r
e
n
t
l
y
 
c
o
d
e
d
 
a
s
 
s
t
r
i
n
g
s
,
 
i
n
t
o
 
d
a
t
e
t
i
m
e
 
f
o
r
m
a
t


d
f
[
'
D
a
t
e
'
]
 
=
 
p
d
.
t
o
_
d
a
t
e
t
i
m
e
(
d
f
[
'
D
a
t
e
'
]
)
```


```python
#
 
e
x
t
r
a
c
t
 
y
e
a
r
 
f
r
o
m
 
d
a
t
e


d
f
[
'
Y
e
a
r
'
]
 
=
 
d
f
[
'
D
a
t
e
'
]
.
d
t
.
y
e
a
r


d
f
[
'
Y
e
a
r
'
]
.
h
e
a
d
(
)
```


```python
#
 
e
x
t
r
a
c
t
 
m
o
n
t
h
 
f
r
o
m
 
d
a
t
e


d
f
[
'
M
o
n
t
h
'
]
 
=
 
d
f
[
'
D
a
t
e
'
]
.
d
t
.
m
o
n
t
h


d
f
[
'
M
o
n
t
h
'
]
.
h
e
a
d
(
)
```


```python
#
 
e
x
t
r
a
c
t
 
d
a
y
 
f
r
o
m
 
d
a
t
e


d
f
[
'
D
a
y
'
]
 
=
 
d
f
[
'
D
a
t
e
'
]
.
d
t
.
d
a
y


d
f
[
'
D
a
y
'
]
.
h
e
a
d
(
)
```


```python
#
 
a
g
a
i
n
 
v
i
e
w
 
t
h
e
 
s
u
m
m
a
r
y
 
o
f
 
d
a
t
a
s
e
t


d
f
.
i
n
f
o
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
t
h
r
e
e
 
a
d
d
i
t
i
o
n
a
l
 
c
o
l
u
m
n
s
 
c
r
e
a
t
e
d
 
f
r
o
m
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
.
 
N
o
w
,
 
I
 
w
i
l
l
 
d
r
o
p
 
t
h
e
 
o
r
i
g
i
n
a
l
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
 
f
r
o
m
 
t
h
e
 
d
a
t
a
s
e
t
.



```python
#
 
d
r
o
p
 
t
h
e
 
o
r
i
g
i
n
a
l
 
D
a
t
e
 
v
a
r
i
a
b
l
e


d
f
.
d
r
o
p
(
'
D
a
t
e
'
,
 
a
x
i
s
=
1
,
 
i
n
p
l
a
c
e
 
=
 
T
r
u
e
)
```


```python
#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
a
g
a
i
n


d
f
.
h
e
a
d
(
)
```

N
o
w
,
 
w
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
 
h
a
s
 
b
e
e
n
 
r
e
m
o
v
e
d
 
f
r
o
m
 
t
h
e
 
d
a
t
a
s
e
t
.




#
#
#
 
E
x
p
l
o
r
e
 
C
a
t
e
g
o
r
i
c
a
l
 
V
a
r
i
a
b
l
e
s






N
o
w
,
 
I
 
w
i
l
l
 
e
x
p
l
o
r
e
 
t
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
o
n
e
 
b
y
 
o
n
e
.
 



```python
#
 
f
i
n
d
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


c
a
t
e
g
o
r
i
c
a
l
 
=
 
[
v
a
r
 
f
o
r
 
v
a
r
 
i
n
 
d
f
.
c
o
l
u
m
n
s
 
i
f
 
d
f
[
v
a
r
]
.
d
t
y
p
e
=
=
'
O
'
]


p
r
i
n
t
(
'
T
h
e
r
e
 
a
r
e
 
{
}
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
\
n
'
.
f
o
r
m
a
t
(
l
e
n
(
c
a
t
e
g
o
r
i
c
a
l
)
)
)


p
r
i
n
t
(
'
T
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
a
r
e
 
:
'
,
 
c
a
t
e
g
o
r
i
c
a
l
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
6
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
h
e
 
d
a
t
a
s
e
t
.
 
T
h
e
 
`
D
a
t
e
`
 
v
a
r
i
a
b
l
e
 
h
a
s
 
b
e
e
n
 
r
e
m
o
v
e
d
.
 
F
i
r
s
t
,
 
I
 
w
i
l
l
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.



```python
#
 
c
h
e
c
k
 
f
o
r
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 


d
f
[
c
a
t
e
g
o
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
`
W
i
n
d
G
u
s
t
D
i
r
`
,
 
`
W
i
n
d
D
i
r
9
a
m
`
,
 
`
W
i
n
d
D
i
r
3
p
m
`
,
 
`
R
a
i
n
T
o
d
a
y
`
 
v
a
r
i
a
b
l
e
s
 
c
o
n
t
a
i
n
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
.
 
I
 
w
i
l
l
 
e
x
p
l
o
r
e
 
t
h
e
s
e
 
v
a
r
i
a
b
l
e
s
 
o
n
e
 
b
y
 
o
n
e
.


#
#
#
 
E
x
p
l
o
r
e
 
`
L
o
c
a
t
i
o
n
`
 
v
a
r
i
a
b
l
e



```python
#
 
p
r
i
n
t
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
i
n
 
L
o
c
a
t
i
o
n
 
v
a
r
i
a
b
l
e


p
r
i
n
t
(
'
L
o
c
a
t
i
o
n
 
c
o
n
t
a
i
n
s
'
,
 
l
e
n
(
d
f
.
L
o
c
a
t
i
o
n
.
u
n
i
q
u
e
(
)
)
,
 
'
l
a
b
e
l
s
'
)
```


```python
#
 
c
h
e
c
k
 
l
a
b
e
l
s
 
i
n
 
l
o
c
a
t
i
o
n
 
v
a
r
i
a
b
l
e


d
f
.
L
o
c
a
t
i
o
n
.
u
n
i
q
u
e
(
)
```


```python
#
 
c
h
e
c
k
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
 
i
n
 
L
o
c
a
t
i
o
n
 
v
a
r
i
a
b
l
e


d
f
.
L
o
c
a
t
i
o
n
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```


```python
#
 
l
e
t
'
s
 
d
o
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 
o
f
 
L
o
c
a
t
i
o
n
 
v
a
r
i
a
b
l
e

#
 
g
e
t
 
k
-
1
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
s
 
a
f
t
e
r
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 

#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
w
i
t
h
 
h
e
a
d
(
)
 
m
e
t
h
o
d


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
L
o
c
a
t
i
o
n
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
)
.
h
e
a
d
(
)
```

#
#
#
 
E
x
p
l
o
r
e
 
`
W
i
n
d
G
u
s
t
D
i
r
`
 
v
a
r
i
a
b
l
e



```python
#
 
p
r
i
n
t
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e


p
r
i
n
t
(
'
W
i
n
d
G
u
s
t
D
i
r
 
c
o
n
t
a
i
n
s
'
,
 
l
e
n
(
d
f
[
'
W
i
n
d
G
u
s
t
D
i
r
'
]
.
u
n
i
q
u
e
(
)
)
,
 
'
l
a
b
e
l
s
'
)
```


```python
#
 
c
h
e
c
k
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e


d
f
[
'
W
i
n
d
G
u
s
t
D
i
r
'
]
.
u
n
i
q
u
e
(
)
```


```python
#
 
c
h
e
c
k
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e


d
f
.
W
i
n
d
G
u
s
t
D
i
r
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```


```python
#
 
l
e
t
'
s
 
d
o
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 
o
f
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e

#
 
g
e
t
 
k
-
1
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
s
 
a
f
t
e
r
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 

#
 
a
l
s
o
 
a
d
d
 
a
n
 
a
d
d
i
t
i
o
n
a
l
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
 
t
o
 
i
n
d
i
c
a
t
e
 
t
h
e
r
e
 
w
a
s
 
m
i
s
s
i
n
g
 
d
a
t
a

#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
w
i
t
h
 
h
e
a
d
(
)
 
m
e
t
h
o
d


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
G
u
s
t
D
i
r
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
h
e
a
d
(
)
```


```python
#
 
s
u
m
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
1
s
 
p
e
r
 
b
o
o
l
e
a
n
 
v
a
r
i
a
b
l
e
 
o
v
e
r
 
t
h
e
 
r
o
w
s
 
o
f
 
t
h
e
 
d
a
t
a
s
e
t

#
 
i
t
 
w
i
l
l
 
t
e
l
l
 
u
s
 
h
o
w
 
m
a
n
y
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
e
 
h
a
v
e
 
f
o
r
 
e
a
c
h
 
c
a
t
e
g
o
r
y


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
G
u
s
t
D
i
r
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
s
u
m
(
a
x
i
s
=
0
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
9
3
3
0
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e
.


#
#
#
 
E
x
p
l
o
r
e
 
`
W
i
n
d
D
i
r
9
a
m
`
 
v
a
r
i
a
b
l
e



```python
#
 
p
r
i
n
t
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
D
i
r
9
a
m
 
v
a
r
i
a
b
l
e


p
r
i
n
t
(
'
W
i
n
d
D
i
r
9
a
m
 
c
o
n
t
a
i
n
s
'
,
 
l
e
n
(
d
f
[
'
W
i
n
d
D
i
r
9
a
m
'
]
.
u
n
i
q
u
e
(
)
)
,
 
'
l
a
b
e
l
s
'
)
```


```python
#
 
c
h
e
c
k
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
D
i
r
9
a
m
 
v
a
r
i
a
b
l
e


d
f
[
'
W
i
n
d
D
i
r
9
a
m
'
]
.
u
n
i
q
u
e
(
)
```


```python
#
 
c
h
e
c
k
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
 
i
n
 
W
i
n
d
D
i
r
9
a
m
 
v
a
r
i
a
b
l
e


d
f
[
'
W
i
n
d
D
i
r
9
a
m
'
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```


```python
#
 
l
e
t
'
s
 
d
o
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 
o
f
 
W
i
n
d
D
i
r
9
a
m
 
v
a
r
i
a
b
l
e

#
 
g
e
t
 
k
-
1
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
s
 
a
f
t
e
r
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 

#
 
a
l
s
o
 
a
d
d
 
a
n
 
a
d
d
i
t
i
o
n
a
l
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
 
t
o
 
i
n
d
i
c
a
t
e
 
t
h
e
r
e
 
w
a
s
 
m
i
s
s
i
n
g
 
d
a
t
a

#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
w
i
t
h
 
h
e
a
d
(
)
 
m
e
t
h
o
d


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
D
i
r
9
a
m
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
h
e
a
d
(
)
```


```python
#
 
s
u
m
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
1
s
 
p
e
r
 
b
o
o
l
e
a
n
 
v
a
r
i
a
b
l
e
 
o
v
e
r
 
t
h
e
 
r
o
w
s
 
o
f
 
t
h
e
 
d
a
t
a
s
e
t

#
 
i
t
 
w
i
l
l
 
t
e
l
l
 
u
s
 
h
o
w
 
m
a
n
y
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
e
 
h
a
v
e
 
f
o
r
 
e
a
c
h
 
c
a
t
e
g
o
r
y


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
D
i
r
9
a
m
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
s
u
m
(
a
x
i
s
=
0
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
1
0
0
1
3
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
`
W
i
n
d
D
i
r
9
a
m
`
 
v
a
r
i
a
b
l
e
.


#
#
#
 
E
x
p
l
o
r
e
 
`
W
i
n
d
D
i
r
3
p
m
`
 
v
a
r
i
a
b
l
e



```python
#
 
p
r
i
n
t
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
D
i
r
3
p
m
 
v
a
r
i
a
b
l
e


p
r
i
n
t
(
'
W
i
n
d
D
i
r
3
p
m
 
c
o
n
t
a
i
n
s
'
,
 
l
e
n
(
d
f
[
'
W
i
n
d
D
i
r
3
p
m
'
]
.
u
n
i
q
u
e
(
)
)
,
 
'
l
a
b
e
l
s
'
)
```


```python
#
 
c
h
e
c
k
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
D
i
r
3
p
m
 
v
a
r
i
a
b
l
e


d
f
[
'
W
i
n
d
D
i
r
3
p
m
'
]
.
u
n
i
q
u
e
(
)
```


```python
#
 
c
h
e
c
k
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
 
i
n
 
W
i
n
d
D
i
r
3
p
m
 
v
a
r
i
a
b
l
e


d
f
[
'
W
i
n
d
D
i
r
3
p
m
'
]
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```


```python
#
 
l
e
t
'
s
 
d
o
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 
o
f
 
W
i
n
d
D
i
r
3
p
m
 
v
a
r
i
a
b
l
e

#
 
g
e
t
 
k
-
1
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
s
 
a
f
t
e
r
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 

#
 
a
l
s
o
 
a
d
d
 
a
n
 
a
d
d
i
t
i
o
n
a
l
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
 
t
o
 
i
n
d
i
c
a
t
e
 
t
h
e
r
e
 
w
a
s
 
m
i
s
s
i
n
g
 
d
a
t
a

#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
w
i
t
h
 
h
e
a
d
(
)
 
m
e
t
h
o
d


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
D
i
r
3
p
m
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
h
e
a
d
(
)
```


```python
#
 
s
u
m
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
1
s
 
p
e
r
 
b
o
o
l
e
a
n
 
v
a
r
i
a
b
l
e
 
o
v
e
r
 
t
h
e
 
r
o
w
s
 
o
f
 
t
h
e
 
d
a
t
a
s
e
t

#
 
i
t
 
w
i
l
l
 
t
e
l
l
 
u
s
 
h
o
w
 
m
a
n
y
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
e
 
h
a
v
e
 
f
o
r
 
e
a
c
h
 
c
a
t
e
g
o
r
y


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
W
i
n
d
D
i
r
3
p
m
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
s
u
m
(
a
x
i
s
=
0
)
```

T
h
e
r
e
 
a
r
e
 
3
7
7
8
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
`
W
i
n
d
D
i
r
3
p
m
`
 
v
a
r
i
a
b
l
e
.


#
#
#
 
E
x
p
l
o
r
e
 
`
R
a
i
n
T
o
d
a
y
`
 
v
a
r
i
a
b
l
e



```python
#
 
p
r
i
n
t
 
n
u
m
b
e
r
 
o
f
 
l
a
b
e
l
s
 
i
n
 
R
a
i
n
T
o
d
a
y
 
v
a
r
i
a
b
l
e


p
r
i
n
t
(
'
R
a
i
n
T
o
d
a
y
 
c
o
n
t
a
i
n
s
'
,
 
l
e
n
(
d
f
[
'
R
a
i
n
T
o
d
a
y
'
]
.
u
n
i
q
u
e
(
)
)
,
 
'
l
a
b
e
l
s
'
)
```


```python
#
 
c
h
e
c
k
 
l
a
b
e
l
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e


d
f
[
'
R
a
i
n
T
o
d
a
y
'
]
.
u
n
i
q
u
e
(
)
```


```python
#
 
c
h
e
c
k
 
f
r
e
q
u
e
n
c
y
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
 
i
n
 
W
i
n
d
G
u
s
t
D
i
r
 
v
a
r
i
a
b
l
e


d
f
.
R
a
i
n
T
o
d
a
y
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```


```python
#
 
l
e
t
'
s
 
d
o
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 
o
f
 
R
a
i
n
T
o
d
a
y
 
v
a
r
i
a
b
l
e

#
 
g
e
t
 
k
-
1
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
s
 
a
f
t
e
r
 
O
n
e
 
H
o
t
 
E
n
c
o
d
i
n
g
 

#
 
a
l
s
o
 
a
d
d
 
a
n
 
a
d
d
i
t
i
o
n
a
l
 
d
u
m
m
y
 
v
a
r
i
a
b
l
e
 
t
o
 
i
n
d
i
c
a
t
e
 
t
h
e
r
e
 
w
a
s
 
m
i
s
s
i
n
g
 
d
a
t
a

#
 
p
r
e
v
i
e
w
 
t
h
e
 
d
a
t
a
s
e
t
 
w
i
t
h
 
h
e
a
d
(
)
 
m
e
t
h
o
d


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
R
a
i
n
T
o
d
a
y
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
h
e
a
d
(
)
```


```python
#
 
s
u
m
 
t
h
e
 
n
u
m
b
e
r
 
o
f
 
1
s
 
p
e
r
 
b
o
o
l
e
a
n
 
v
a
r
i
a
b
l
e
 
o
v
e
r
 
t
h
e
 
r
o
w
s
 
o
f
 
t
h
e
 
d
a
t
a
s
e
t

#
 
i
t
 
w
i
l
l
 
t
e
l
l
 
u
s
 
h
o
w
 
m
a
n
y
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
e
 
h
a
v
e
 
f
o
r
 
e
a
c
h
 
c
a
t
e
g
o
r
y


p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
d
f
.
R
a
i
n
T
o
d
a
y
,
 
d
r
o
p
_
f
i
r
s
t
=
T
r
u
e
,
 
d
u
m
m
y
_
n
a
=
T
r
u
e
)
.
s
u
m
(
a
x
i
s
=
0
)
```

T
h
e
r
e
 
a
r
e
 
1
4
0
6
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
`
R
a
i
n
T
o
d
a
y
`
 
v
a
r
i
a
b
l
e
.


#
#
#
 
E
x
p
l
o
r
e
 
N
u
m
e
r
i
c
a
l
 
V
a
r
i
a
b
l
e
s



```python
#
 
f
i
n
d
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


n
u
m
e
r
i
c
a
l
 
=
 
[
v
a
r
 
f
o
r
 
v
a
r
 
i
n
 
d
f
.
c
o
l
u
m
n
s
 
i
f
 
d
f
[
v
a
r
]
.
d
t
y
p
e
!
=
'
O
'
]


p
r
i
n
t
(
'
T
h
e
r
e
 
a
r
e
 
{
}
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
\
n
'
.
f
o
r
m
a
t
(
l
e
n
(
n
u
m
e
r
i
c
a
l
)
)
)


p
r
i
n
t
(
'
T
h
e
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
a
r
e
 
:
'
,
 
n
u
m
e
r
i
c
a
l
)
```


```python
#
 
v
i
e
w
 
t
h
e
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


d
f
[
n
u
m
e
r
i
c
a
l
]
.
h
e
a
d
(
)
```

#
#
#
 
S
u
m
m
a
r
y
 
o
f
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






-
 
T
h
e
r
e
 
a
r
e
 
1
6
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.
 






-
 
T
h
e
s
e
 
a
r
e
 
g
i
v
e
n
 
b
y
 
`
M
i
n
T
e
m
p
`
,
 
`
M
a
x
T
e
m
p
`
,
 
`
R
a
i
n
f
a
l
l
`
,
 
`
E
v
a
p
o
r
a
t
i
o
n
`
,
 
`
S
u
n
s
h
i
n
e
`
,
 
`
W
i
n
d
G
u
s
t
S
p
e
e
d
`
,
 
`
W
i
n
d
S
p
e
e
d
9
a
m
`
,
 
`
W
i
n
d
S
p
e
e
d
3
p
m
`
,
 
`
H
u
m
i
d
i
t
y
9
a
m
`
,
 
`
H
u
m
i
d
i
t
y
3
p
m
`
,
 
`
P
r
e
s
s
u
r
e
9
a
m
`
,
 
`
P
r
e
s
s
u
r
e
3
p
m
`
,
 
`
C
l
o
u
d
9
a
m
`
,
 
`
C
l
o
u
d
3
p
m
`
,
 
`
T
e
m
p
9
a
m
`
 
a
n
d
 
`
T
e
m
p
3
p
m
`
.






-
 
A
l
l
 
o
f
 
t
h
e
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
a
r
e
 
o
f
 
c
o
n
t
i
n
u
o
u
s
 
t
y
p
e
.


#
#
 
E
x
p
l
o
r
e
 
p
r
o
b
l
e
m
s
 
w
i
t
h
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






N
o
w
,
 
I
 
w
i
l
l
 
e
x
p
l
o
r
e
 
t
h
e
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
.






#
#
#
 
M
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s



```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


d
f
[
n
u
m
e
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
a
l
l
 
t
h
e
 
1
6
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
c
o
n
t
a
i
n
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
.


#
#
#
 
O
u
t
l
i
e
r
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s



```python
#
 
v
i
e
w
 
s
u
m
m
a
r
y
 
s
t
a
t
i
s
t
i
c
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


p
r
i
n
t
(
r
o
u
n
d
(
d
f
[
n
u
m
e
r
i
c
a
l
]
.
d
e
s
c
r
i
b
e
(
)
)
,
2
)
```

O
n
 
c
l
o
s
e
r
 
i
n
s
p
e
c
t
i
o
n
,
 
w
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
 
`
R
a
i
n
f
a
l
l
`
,
 
`
E
v
a
p
o
r
a
t
i
o
n
`
,
 
`
W
i
n
d
S
p
e
e
d
9
a
m
`
 
a
n
d
 
`
W
i
n
d
S
p
e
e
d
3
p
m
`
 
c
o
l
u
m
n
s
 
m
a
y
 
c
o
n
t
a
i
n
 
o
u
t
l
i
e
r
s
.






I
 
w
i
l
l
 
d
r
a
w
 
b
o
x
p
l
o
t
s
 
t
o
 
v
i
s
u
a
l
i
s
e
 
o
u
t
l
i
e
r
s
 
i
n
 
t
h
e
 
a
b
o
v
e
 
v
a
r
i
a
b
l
e
s
.
 



```python
#
 
d
r
a
w
 
b
o
x
p
l
o
t
s
 
t
o
 
v
i
s
u
a
l
i
z
e
 
o
u
t
l
i
e
r
s


p
l
t
.
f
i
g
u
r
e
(
f
i
g
s
i
z
e
=
(
1
5
,
1
0
)
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
1
)

f
i
g
 
=
 
d
f
.
b
o
x
p
l
o
t
(
c
o
l
u
m
n
=
'
R
a
i
n
f
a
l
l
'
)

f
i
g
.
s
e
t
_
t
i
t
l
e
(
'
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
R
a
i
n
f
a
l
l
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
2
)

f
i
g
 
=
 
d
f
.
b
o
x
p
l
o
t
(
c
o
l
u
m
n
=
'
E
v
a
p
o
r
a
t
i
o
n
'
)

f
i
g
.
s
e
t
_
t
i
t
l
e
(
'
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
E
v
a
p
o
r
a
t
i
o
n
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
3
)

f
i
g
 
=
 
d
f
.
b
o
x
p
l
o
t
(
c
o
l
u
m
n
=
'
W
i
n
d
S
p
e
e
d
9
a
m
'
)

f
i
g
.
s
e
t
_
t
i
t
l
e
(
'
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
W
i
n
d
S
p
e
e
d
9
a
m
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
4
)

f
i
g
 
=
 
d
f
.
b
o
x
p
l
o
t
(
c
o
l
u
m
n
=
'
W
i
n
d
S
p
e
e
d
3
p
m
'
)

f
i
g
.
s
e
t
_
t
i
t
l
e
(
'
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
W
i
n
d
S
p
e
e
d
3
p
m
'
)
```

T
h
e
 
a
b
o
v
e
 
b
o
x
p
l
o
t
s
 
c
o
n
f
i
r
m
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
l
o
t
 
o
f
 
o
u
t
l
i
e
r
s
 
i
n
 
t
h
e
s
e
 
v
a
r
i
a
b
l
e
s
.


#
#
#
 
C
h
e
c
k
 
t
h
e
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
r
i
a
b
l
e
s






N
o
w
,
 
I
 
w
i
l
l
 
p
l
o
t
 
t
h
e
 
h
i
s
t
o
g
r
a
m
s
 
t
o
 
c
h
e
c
k
 
d
i
s
t
r
i
b
u
t
i
o
n
s
 
t
o
 
f
i
n
d
 
o
u
t
 
i
f
 
t
h
e
y
 
a
r
e
 
n
o
r
m
a
l
 
o
r
 
s
k
e
w
e
d
.
 
I
f
 
t
h
e
 
v
a
r
i
a
b
l
e
 
f
o
l
l
o
w
s
 
n
o
r
m
a
l
 
d
i
s
t
r
i
b
u
t
i
o
n
,
 
t
h
e
n
 
I
 
w
i
l
l
 
d
o
 
`
E
x
t
r
e
m
e
 
V
a
l
u
e
 
A
n
a
l
y
s
i
s
`
 
o
t
h
e
r
w
i
s
e
 
i
f
 
t
h
e
y
 
a
r
e
 
s
k
e
w
e
d
,
 
I
 
w
i
l
l
 
f
i
n
d
 
I
Q
R
 
(
I
n
t
e
r
q
u
a
n
t
i
l
e
 
r
a
n
g
e
)
.



```python
#
 
p
l
o
t
 
h
i
s
t
o
g
r
a
m
 
t
o
 
c
h
e
c
k
 
d
i
s
t
r
i
b
u
t
i
o
n


p
l
t
.
f
i
g
u
r
e
(
f
i
g
s
i
z
e
=
(
1
5
,
1
0
)
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
1
)

f
i
g
 
=
 
d
f
.
R
a
i
n
f
a
l
l
.
h
i
s
t
(
b
i
n
s
=
1
0
)

f
i
g
.
s
e
t
_
x
l
a
b
e
l
(
'
R
a
i
n
f
a
l
l
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
R
a
i
n
T
o
m
o
r
r
o
w
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
2
)

f
i
g
 
=
 
d
f
.
E
v
a
p
o
r
a
t
i
o
n
.
h
i
s
t
(
b
i
n
s
=
1
0
)

f
i
g
.
s
e
t
_
x
l
a
b
e
l
(
'
E
v
a
p
o
r
a
t
i
o
n
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
R
a
i
n
T
o
m
o
r
r
o
w
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
3
)

f
i
g
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
9
a
m
.
h
i
s
t
(
b
i
n
s
=
1
0
)

f
i
g
.
s
e
t
_
x
l
a
b
e
l
(
'
W
i
n
d
S
p
e
e
d
9
a
m
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
R
a
i
n
T
o
m
o
r
r
o
w
'
)



p
l
t
.
s
u
b
p
l
o
t
(
2
,
 
2
,
 
4
)

f
i
g
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
3
p
m
.
h
i
s
t
(
b
i
n
s
=
1
0
)

f
i
g
.
s
e
t
_
x
l
a
b
e
l
(
'
W
i
n
d
S
p
e
e
d
3
p
m
'
)

f
i
g
.
s
e
t
_
y
l
a
b
e
l
(
'
R
a
i
n
T
o
m
o
r
r
o
w
'
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
a
l
l
 
t
h
e
 
f
o
u
r
 
v
a
r
i
a
b
l
e
s
 
a
r
e
 
s
k
e
w
e
d
.
 
S
o
,
 
I
 
w
i
l
l
 
u
s
e
 
i
n
t
e
r
q
u
a
n
t
i
l
e
 
r
a
n
g
e
 
t
o
 
f
i
n
d
 
o
u
t
l
i
e
r
s
.



```python
#
 
f
i
n
d
 
o
u
t
l
i
e
r
s
 
f
o
r
 
R
a
i
n
f
a
l
l
 
v
a
r
i
a
b
l
e


I
Q
R
 
=
 
d
f
.
R
a
i
n
f
a
l
l
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
-
 
d
f
.
R
a
i
n
f
a
l
l
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)

L
o
w
e
r
_
f
e
n
c
e
 
=
 
d
f
.
R
a
i
n
f
a
l
l
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)
 
-
 
(
I
Q
R
 
*
 
3
)

U
p
p
e
r
_
f
e
n
c
e
 
=
 
d
f
.
R
a
i
n
f
a
l
l
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
+
 
(
I
Q
R
 
*
 
3
)

p
r
i
n
t
(
'
R
a
i
n
f
a
l
l
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
<
 
{
l
o
w
e
r
b
o
u
n
d
a
r
y
}
 
o
r
 
>
 
{
u
p
p
e
r
b
o
u
n
d
a
r
y
}
'
.
f
o
r
m
a
t
(
l
o
w
e
r
b
o
u
n
d
a
r
y
=
L
o
w
e
r
_
f
e
n
c
e
,
 
u
p
p
e
r
b
o
u
n
d
a
r
y
=
U
p
p
e
r
_
f
e
n
c
e
)
)

```

F
o
r
 
`
R
a
i
n
f
a
l
l
`
,
 
t
h
e
 
m
i
n
i
m
u
m
 
a
n
d
 
m
a
x
i
m
u
m
 
v
a
l
u
e
s
 
a
r
e
 
0
.
0
 
a
n
d
 
3
7
1
.
0
.
 
S
o
,
 
t
h
e
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
>
 
3
.
2
.



```python
#
 
f
i
n
d
 
o
u
t
l
i
e
r
s
 
f
o
r
 
E
v
a
p
o
r
a
t
i
o
n
 
v
a
r
i
a
b
l
e


I
Q
R
 
=
 
d
f
.
E
v
a
p
o
r
a
t
i
o
n
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
-
 
d
f
.
E
v
a
p
o
r
a
t
i
o
n
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)

L
o
w
e
r
_
f
e
n
c
e
 
=
 
d
f
.
E
v
a
p
o
r
a
t
i
o
n
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)
 
-
 
(
I
Q
R
 
*
 
3
)

U
p
p
e
r
_
f
e
n
c
e
 
=
 
d
f
.
E
v
a
p
o
r
a
t
i
o
n
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
+
 
(
I
Q
R
 
*
 
3
)

p
r
i
n
t
(
'
E
v
a
p
o
r
a
t
i
o
n
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
<
 
{
l
o
w
e
r
b
o
u
n
d
a
r
y
}
 
o
r
 
>
 
{
u
p
p
e
r
b
o
u
n
d
a
r
y
}
'
.
f
o
r
m
a
t
(
l
o
w
e
r
b
o
u
n
d
a
r
y
=
L
o
w
e
r
_
f
e
n
c
e
,
 
u
p
p
e
r
b
o
u
n
d
a
r
y
=
U
p
p
e
r
_
f
e
n
c
e
)
)

```

F
o
r
 
`
E
v
a
p
o
r
a
t
i
o
n
`
,
 
t
h
e
 
m
i
n
i
m
u
m
 
a
n
d
 
m
a
x
i
m
u
m
 
v
a
l
u
e
s
 
a
r
e
 
0
.
0
 
a
n
d
 
1
4
5
.
0
.
 
S
o
,
 
t
h
e
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
>
 
2
1
.
8
.



```python
#
 
f
i
n
d
 
o
u
t
l
i
e
r
s
 
f
o
r
 
W
i
n
d
S
p
e
e
d
9
a
m
 
v
a
r
i
a
b
l
e


I
Q
R
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
9
a
m
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
-
 
d
f
.
W
i
n
d
S
p
e
e
d
9
a
m
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)

L
o
w
e
r
_
f
e
n
c
e
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
9
a
m
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)
 
-
 
(
I
Q
R
 
*
 
3
)

U
p
p
e
r
_
f
e
n
c
e
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
9
a
m
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
+
 
(
I
Q
R
 
*
 
3
)

p
r
i
n
t
(
'
W
i
n
d
S
p
e
e
d
9
a
m
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
<
 
{
l
o
w
e
r
b
o
u
n
d
a
r
y
}
 
o
r
 
>
 
{
u
p
p
e
r
b
o
u
n
d
a
r
y
}
'
.
f
o
r
m
a
t
(
l
o
w
e
r
b
o
u
n
d
a
r
y
=
L
o
w
e
r
_
f
e
n
c
e
,
 
u
p
p
e
r
b
o
u
n
d
a
r
y
=
U
p
p
e
r
_
f
e
n
c
e
)
)

```

F
o
r
 
`
W
i
n
d
S
p
e
e
d
9
a
m
`
,
 
t
h
e
 
m
i
n
i
m
u
m
 
a
n
d
 
m
a
x
i
m
u
m
 
v
a
l
u
e
s
 
a
r
e
 
0
.
0
 
a
n
d
 
1
3
0
.
0
.
 
S
o
,
 
t
h
e
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
>
 
5
5
.
0
.



```python
#
 
f
i
n
d
 
o
u
t
l
i
e
r
s
 
f
o
r
 
W
i
n
d
S
p
e
e
d
3
p
m
 
v
a
r
i
a
b
l
e


I
Q
R
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
3
p
m
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
-
 
d
f
.
W
i
n
d
S
p
e
e
d
3
p
m
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)

L
o
w
e
r
_
f
e
n
c
e
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
3
p
m
.
q
u
a
n
t
i
l
e
(
0
.
2
5
)
 
-
 
(
I
Q
R
 
*
 
3
)

U
p
p
e
r
_
f
e
n
c
e
 
=
 
d
f
.
W
i
n
d
S
p
e
e
d
3
p
m
.
q
u
a
n
t
i
l
e
(
0
.
7
5
)
 
+
 
(
I
Q
R
 
*
 
3
)

p
r
i
n
t
(
'
W
i
n
d
S
p
e
e
d
3
p
m
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
<
 
{
l
o
w
e
r
b
o
u
n
d
a
r
y
}
 
o
r
 
>
 
{
u
p
p
e
r
b
o
u
n
d
a
r
y
}
'
.
f
o
r
m
a
t
(
l
o
w
e
r
b
o
u
n
d
a
r
y
=
L
o
w
e
r
_
f
e
n
c
e
,
 
u
p
p
e
r
b
o
u
n
d
a
r
y
=
U
p
p
e
r
_
f
e
n
c
e
)
)

```

F
o
r
 
`
W
i
n
d
S
p
e
e
d
3
p
m
`
,
 
t
h
e
 
m
i
n
i
m
u
m
 
a
n
d
 
m
a
x
i
m
u
m
 
v
a
l
u
e
s
 
a
r
e
 
0
.
0
 
a
n
d
 
8
7
.
0
.
 
S
o
,
 
t
h
e
 
o
u
t
l
i
e
r
s
 
a
r
e
 
v
a
l
u
e
s
 
>
 
5
7
.
0
.


#
 
*
*
8
.
 
D
e
c
l
a
r
e
 
f
e
a
t
u
r
e
 
v
e
c
t
o
r
 
a
n
d
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
8
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
X
 
=
 
d
f
.
d
r
o
p
(
[
'
R
a
i
n
T
o
m
o
r
r
o
w
'
]
,
 
a
x
i
s
=
1
)


y
 
=
 
d
f
[
'
R
a
i
n
T
o
m
o
r
r
o
w
'
]
```

#
 
*
*
9
.
 
S
p
l
i
t
 
d
a
t
a
 
i
n
t
o
 
s
e
p
a
r
a
t
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
9
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
#
 
s
p
l
i
t
 
X
 
a
n
d
 
y
 
i
n
t
o
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
i
n
g
 
s
e
t
s


f
r
o
m
 
s
k
l
e
a
r
n
.
m
o
d
e
l
_
s
e
l
e
c
t
i
o
n
 
i
m
p
o
r
t
 
t
r
a
i
n
_
t
e
s
t
_
s
p
l
i
t


X
_
t
r
a
i
n
,
 
X
_
t
e
s
t
,
 
y
_
t
r
a
i
n
,
 
y
_
t
e
s
t
 
=
 
t
r
a
i
n
_
t
e
s
t
_
s
p
l
i
t
(
X
,
 
y
,
 
t
e
s
t
_
s
i
z
e
 
=
 
0
.
2
,
 
r
a
n
d
o
m
_
s
t
a
t
e
 
=
 
0
)

```


```python
#
 
c
h
e
c
k
 
t
h
e
 
s
h
a
p
e
 
o
f
 
X
_
t
r
a
i
n
 
a
n
d
 
X
_
t
e
s
t


X
_
t
r
a
i
n
.
s
h
a
p
e
,
 
X
_
t
e
s
t
.
s
h
a
p
e
```

#
 
*
*
1
0
.
 
F
e
a
t
u
r
e
 
E
n
g
i
n
e
e
r
i
n
g
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
0
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






*
*
F
e
a
t
u
r
e
 
E
n
g
i
n
e
e
r
i
n
g
*
*
 
i
s
 
t
h
e
 
p
r
o
c
e
s
s
 
o
f
 
t
r
a
n
s
f
o
r
m
i
n
g
 
r
a
w
 
d
a
t
a
 
i
n
t
o
 
u
s
e
f
u
l
 
f
e
a
t
u
r
e
s
 
t
h
a
t
 
h
e
l
p
 
u
s
 
t
o
 
u
n
d
e
r
s
t
a
n
d
 
o
u
r
 
m
o
d
e
l
 
b
e
t
t
e
r
 
a
n
d
 
i
n
c
r
e
a
s
e
 
i
t
s
 
p
r
e
d
i
c
t
i
v
e
 
p
o
w
e
r
.
 
I
 
w
i
l
l
 
c
a
r
r
y
 
o
u
t
 
f
e
a
t
u
r
e
 
e
n
g
i
n
e
e
r
i
n
g
 
o
n
 
d
i
f
f
e
r
e
n
t
 
t
y
p
e
s
 
o
f
 
v
a
r
i
a
b
l
e
s
.






F
i
r
s
t
,
 
I
 
w
i
l
l
 
d
i
s
p
l
a
y
 
t
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
a
n
d
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
a
g
a
i
n
 
s
e
p
a
r
a
t
e
l
y
.



```python
#
 
c
h
e
c
k
 
d
a
t
a
 
t
y
p
e
s
 
i
n
 
X
_
t
r
a
i
n


X
_
t
r
a
i
n
.
d
t
y
p
e
s
```


```python
#
 
d
i
s
p
l
a
y
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


c
a
t
e
g
o
r
i
c
a
l
 
=
 
[
c
o
l
 
f
o
r
 
c
o
l
 
i
n
 
X
_
t
r
a
i
n
.
c
o
l
u
m
n
s
 
i
f
 
X
_
t
r
a
i
n
[
c
o
l
]
.
d
t
y
p
e
s
 
=
=
 
'
O
'
]


c
a
t
e
g
o
r
i
c
a
l
```


```python
#
 
d
i
s
p
l
a
y
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s


n
u
m
e
r
i
c
a
l
 
=
 
[
c
o
l
 
f
o
r
 
c
o
l
 
i
n
 
X
_
t
r
a
i
n
.
c
o
l
u
m
n
s
 
i
f
 
X
_
t
r
a
i
n
[
c
o
l
]
.
d
t
y
p
e
s
 
!
=
 
'
O
'
]


n
u
m
e
r
i
c
a
l
```

#
#
#
 
E
n
g
i
n
e
e
r
i
n
g
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s







```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
r
a
i
n


X
_
t
r
a
i
n
[
n
u
m
e
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
e
s
t


X
_
t
e
s
t
[
n
u
m
e
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
p
r
i
n
t
 
p
e
r
c
e
n
t
a
g
e
 
o
f
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
r
a
i
n
i
n
g
 
s
e
t


f
o
r
 
c
o
l
 
i
n
 
n
u
m
e
r
i
c
a
l
:

 
 
 
 
i
f
 
X
_
t
r
a
i
n
[
c
o
l
]
.
i
s
n
u
l
l
(
)
.
m
e
a
n
(
)
>
0
:

 
 
 
 
 
 
 
 
p
r
i
n
t
(
c
o
l
,
 
r
o
u
n
d
(
X
_
t
r
a
i
n
[
c
o
l
]
.
i
s
n
u
l
l
(
)
.
m
e
a
n
(
)
,
4
)
)
```

#
#
#
 
A
s
s
u
m
p
t
i
o
n






I
 
a
s
s
u
m
e
 
t
h
a
t
 
t
h
e
 
d
a
t
a
 
a
r
e
 
m
i
s
s
i
n
g
 
c
o
m
p
l
e
t
e
l
y
 
a
t
 
r
a
n
d
o
m
 
(
M
C
A
R
)
.
 
T
h
e
r
e
 
a
r
e
 
t
w
o
 
m
e
t
h
o
d
s
 
w
h
i
c
h
 
c
a
n
 
b
e
 
u
s
e
d
 
t
o
 
i
m
p
u
t
e
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
.
 
O
n
e
 
i
s
 
m
e
a
n
 
o
r
 
m
e
d
i
a
n
 
i
m
p
u
t
a
t
i
o
n
 
a
n
d
 
o
t
h
e
r
 
o
n
e
 
i
s
 
r
a
n
d
o
m
 
s
a
m
p
l
e
 
i
m
p
u
t
a
t
i
o
n
.
 
W
h
e
n
 
t
h
e
r
e
 
a
r
e
 
o
u
t
l
i
e
r
s
 
i
n
 
t
h
e
 
d
a
t
a
s
e
t
,
 
w
e
 
s
h
o
u
l
d
 
u
s
e
 
m
e
d
i
a
n
 
i
m
p
u
t
a
t
i
o
n
.
 
S
o
,
 
I
 
w
i
l
l
 
u
s
e
 
m
e
d
i
a
n
 
i
m
p
u
t
a
t
i
o
n
 
b
e
c
a
u
s
e
 
m
e
d
i
a
n
 
i
m
p
u
t
a
t
i
o
n
 
i
s
 
r
o
b
u
s
t
 
t
o
 
o
u
t
l
i
e
r
s
.






I
 
w
i
l
l
 
i
m
p
u
t
e
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
w
i
t
h
 
t
h
e
 
a
p
p
r
o
p
r
i
a
t
e
 
s
t
a
t
i
s
t
i
c
a
l
 
m
e
a
s
u
r
e
s
 
o
f
 
t
h
e
 
d
a
t
a
,
 
i
n
 
t
h
i
s
 
c
a
s
e
 
m
e
d
i
a
n
.
 
I
m
p
u
t
a
t
i
o
n
 
s
h
o
u
l
d
 
b
e
 
d
o
n
e
 
o
v
e
r
 
t
h
e
 
t
r
a
i
n
i
n
g
 
s
e
t
,
 
a
n
d
 
t
h
e
n
 
p
r
o
p
a
g
a
t
e
d
 
t
o
 
t
h
e
 
t
e
s
t
 
s
e
t
.
 
I
t
 
m
e
a
n
s
 
t
h
a
t
 
t
h
e
 
s
t
a
t
i
s
t
i
c
a
l
 
m
e
a
s
u
r
e
s
 
t
o
 
b
e
 
u
s
e
d
 
t
o
 
f
i
l
l
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
b
o
t
h
 
i
n
 
t
r
a
i
n
 
a
n
d
 
t
e
s
t
 
s
e
t
,
 
s
h
o
u
l
d
 
b
e
 
e
x
t
r
a
c
t
e
d
 
f
r
o
m
 
t
h
e
 
t
r
a
i
n
 
s
e
t
 
o
n
l
y
.
 
T
h
i
s
 
i
s
 
t
o
 
a
v
o
i
d
 
o
v
e
r
f
i
t
t
i
n
g
.



```python
#
 
i
m
p
u
t
e
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
X
_
t
r
a
i
n
 
a
n
d
 
X
_
t
e
s
t
 
w
i
t
h
 
r
e
s
p
e
c
t
i
v
e
 
c
o
l
u
m
n
 
m
e
d
i
a
n
 
i
n
 
X
_
t
r
a
i
n


f
o
r
 
d
f
1
 
i
n
 
[
X
_
t
r
a
i
n
,
 
X
_
t
e
s
t
]
:

 
 
 
 
f
o
r
 
c
o
l
 
i
n
 
n
u
m
e
r
i
c
a
l
:

 
 
 
 
 
 
 
 
c
o
l
_
m
e
d
i
a
n
=
X
_
t
r
a
i
n
[
c
o
l
]
.
m
e
d
i
a
n
(
)

 
 
 
 
 
 
 
 
d
f
1
[
c
o
l
]
.
f
i
l
l
n
a
(
c
o
l
_
m
e
d
i
a
n
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)
 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
```


```python
#
 
c
h
e
c
k
 
a
g
a
i
n
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
r
a
i
n


X
_
t
r
a
i
n
[
n
u
m
e
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
e
s
t


X
_
t
e
s
t
[
n
u
m
e
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

N
o
w
,
 
w
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
n
o
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
n
u
m
e
r
i
c
a
l
 
c
o
l
u
m
n
s
 
o
f
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t
.


#
#
#
 
E
n
g
i
n
e
e
r
i
n
g
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s



```python
#
 
p
r
i
n
t
 
p
e
r
c
e
n
t
a
g
e
 
o
f
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
t
h
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
t
r
a
i
n
i
n
g
 
s
e
t


X
_
t
r
a
i
n
[
c
a
t
e
g
o
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
m
e
a
n
(
)
```


```python
#
 
p
r
i
n
t
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
w
i
t
h
 
m
i
s
s
i
n
g
 
d
a
t
a


f
o
r
 
c
o
l
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
:

 
 
 
 
i
f
 
X
_
t
r
a
i
n
[
c
o
l
]
.
i
s
n
u
l
l
(
)
.
m
e
a
n
(
)
>
0
:

 
 
 
 
 
 
 
 
p
r
i
n
t
(
c
o
l
,
 
(
X
_
t
r
a
i
n
[
c
o
l
]
.
i
s
n
u
l
l
(
)
.
m
e
a
n
(
)
)
)
```


```python
#
 
i
m
p
u
t
e
 
m
i
s
s
i
n
g
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
w
i
t
h
 
m
o
s
t
 
f
r
e
q
u
e
n
t
 
v
a
l
u
e


f
o
r
 
d
f
2
 
i
n
 
[
X
_
t
r
a
i
n
,
 
X
_
t
e
s
t
]
:

 
 
 
 
d
f
2
[
'
W
i
n
d
G
u
s
t
D
i
r
'
]
.
f
i
l
l
n
a
(
X
_
t
r
a
i
n
[
'
W
i
n
d
G
u
s
t
D
i
r
'
]
.
m
o
d
e
(
)
[
0
]
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)

 
 
 
 
d
f
2
[
'
W
i
n
d
D
i
r
9
a
m
'
]
.
f
i
l
l
n
a
(
X
_
t
r
a
i
n
[
'
W
i
n
d
D
i
r
9
a
m
'
]
.
m
o
d
e
(
)
[
0
]
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)

 
 
 
 
d
f
2
[
'
W
i
n
d
D
i
r
3
p
m
'
]
.
f
i
l
l
n
a
(
X
_
t
r
a
i
n
[
'
W
i
n
d
D
i
r
3
p
m
'
]
.
m
o
d
e
(
)
[
0
]
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)

 
 
 
 
d
f
2
[
'
R
a
i
n
T
o
d
a
y
'
]
.
f
i
l
l
n
a
(
X
_
t
r
a
i
n
[
'
R
a
i
n
T
o
d
a
y
'
]
.
m
o
d
e
(
)
[
0
]
,
 
i
n
p
l
a
c
e
=
T
r
u
e
)
```


```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
r
a
i
n


X
_
t
r
a
i
n
[
c
a
t
e
g
o
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s
 
i
n
 
X
_
t
e
s
t


X
_
t
e
s
t
[
c
a
t
e
g
o
r
i
c
a
l
]
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

A
s
 
a
 
f
i
n
a
l
 
c
h
e
c
k
,
 
I
 
w
i
l
l
 
c
h
e
c
k
 
f
o
r
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
X
_
t
r
a
i
n
 
a
n
d
 
X
_
t
e
s
t
.



```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
X
_
t
r
a
i
n


X
_
t
r
a
i
n
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```


```python
#
 
c
h
e
c
k
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
X
_
t
e
s
t


X
_
t
e
s
t
.
i
s
n
u
l
l
(
)
.
s
u
m
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
n
o
 
m
i
s
s
i
n
g
 
v
a
l
u
e
s
 
i
n
 
X
_
t
r
a
i
n
 
a
n
d
 
X
_
t
e
s
t
.


#
#
#
 
E
n
g
i
n
e
e
r
i
n
g
 
o
u
t
l
i
e
r
s
 
i
n
 
n
u
m
e
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s






W
e
 
h
a
v
e
 
s
e
e
n
 
t
h
a
t
 
t
h
e
 
`
R
a
i
n
f
a
l
l
`
,
 
`
E
v
a
p
o
r
a
t
i
o
n
`
,
 
`
W
i
n
d
S
p
e
e
d
9
a
m
`
 
a
n
d
 
`
W
i
n
d
S
p
e
e
d
3
p
m
`
 
c
o
l
u
m
n
s
 
c
o
n
t
a
i
n
 
o
u
t
l
i
e
r
s
.
 
I
 
w
i
l
l
 
u
s
e
 
t
o
p
-
c
o
d
i
n
g
 
a
p
p
r
o
a
c
h
 
t
o
 
c
a
p
 
m
a
x
i
m
u
m
 
v
a
l
u
e
s
 
a
n
d
 
r
e
m
o
v
e
 
o
u
t
l
i
e
r
s
 
f
r
o
m
 
t
h
e
 
a
b
o
v
e
 
v
a
r
i
a
b
l
e
s
.



```python
d
e
f
 
m
a
x
_
v
a
l
u
e
(
d
f
3
,
 
v
a
r
i
a
b
l
e
,
 
t
o
p
)
:

 
 
 
 
r
e
t
u
r
n
 
n
p
.
w
h
e
r
e
(
d
f
3
[
v
a
r
i
a
b
l
e
]
>
t
o
p
,
 
t
o
p
,
 
d
f
3
[
v
a
r
i
a
b
l
e
]
)


f
o
r
 
d
f
3
 
i
n
 
[
X
_
t
r
a
i
n
,
 
X
_
t
e
s
t
]
:

 
 
 
 
d
f
3
[
'
R
a
i
n
f
a
l
l
'
]
 
=
 
m
a
x
_
v
a
l
u
e
(
d
f
3
,
 
'
R
a
i
n
f
a
l
l
'
,
 
3
.
2
)

 
 
 
 
d
f
3
[
'
E
v
a
p
o
r
a
t
i
o
n
'
]
 
=
 
m
a
x
_
v
a
l
u
e
(
d
f
3
,
 
'
E
v
a
p
o
r
a
t
i
o
n
'
,
 
2
1
.
8
)

 
 
 
 
d
f
3
[
'
W
i
n
d
S
p
e
e
d
9
a
m
'
]
 
=
 
m
a
x
_
v
a
l
u
e
(
d
f
3
,
 
'
W
i
n
d
S
p
e
e
d
9
a
m
'
,
 
5
5
)

 
 
 
 
d
f
3
[
'
W
i
n
d
S
p
e
e
d
3
p
m
'
]
 
=
 
m
a
x
_
v
a
l
u
e
(
d
f
3
,
 
'
W
i
n
d
S
p
e
e
d
3
p
m
'
,
 
5
7
)
```


```python
X
_
t
r
a
i
n
.
R
a
i
n
f
a
l
l
.
m
a
x
(
)
,
 
X
_
t
e
s
t
.
R
a
i
n
f
a
l
l
.
m
a
x
(
)
```


```python
X
_
t
r
a
i
n
.
E
v
a
p
o
r
a
t
i
o
n
.
m
a
x
(
)
,
 
X
_
t
e
s
t
.
E
v
a
p
o
r
a
t
i
o
n
.
m
a
x
(
)
```


```python
X
_
t
r
a
i
n
.
W
i
n
d
S
p
e
e
d
9
a
m
.
m
a
x
(
)
,
 
X
_
t
e
s
t
.
W
i
n
d
S
p
e
e
d
9
a
m
.
m
a
x
(
)
```


```python
X
_
t
r
a
i
n
.
W
i
n
d
S
p
e
e
d
3
p
m
.
m
a
x
(
)
,
 
X
_
t
e
s
t
.
W
i
n
d
S
p
e
e
d
3
p
m
.
m
a
x
(
)
```


```python
X
_
t
r
a
i
n
[
n
u
m
e
r
i
c
a
l
]
.
d
e
s
c
r
i
b
e
(
)
```

W
e
 
c
a
n
 
n
o
w
 
s
e
e
 
t
h
a
t
 
t
h
e
 
o
u
t
l
i
e
r
s
 
i
n
 
`
R
a
i
n
f
a
l
l
`
,
 
`
E
v
a
p
o
r
a
t
i
o
n
`
,
 
`
W
i
n
d
S
p
e
e
d
9
a
m
`
 
a
n
d
 
`
W
i
n
d
S
p
e
e
d
3
p
m
`
 
c
o
l
u
m
n
s
 
a
r
e
 
c
a
p
p
e
d
.


#
#
#
 
E
n
c
o
d
e
 
c
a
t
e
g
o
r
i
c
a
l
 
v
a
r
i
a
b
l
e
s



```python
c
a
t
e
g
o
r
i
c
a
l
```


```python
X
_
t
r
a
i
n
[
c
a
t
e
g
o
r
i
c
a
l
]
.
h
e
a
d
(
)
```


```python
#
 
e
n
c
o
d
e
 
R
a
i
n
T
o
d
a
y
 
v
a
r
i
a
b
l
e


i
m
p
o
r
t
 
c
a
t
e
g
o
r
y
_
e
n
c
o
d
e
r
s
 
a
s
 
c
e


e
n
c
o
d
e
r
 
=
 
c
e
.
B
i
n
a
r
y
E
n
c
o
d
e
r
(
c
o
l
s
=
[
'
R
a
i
n
T
o
d
a
y
'
]
)


X
_
t
r
a
i
n
 
=
 
e
n
c
o
d
e
r
.
f
i
t
_
t
r
a
n
s
f
o
r
m
(
X
_
t
r
a
i
n
)


X
_
t
e
s
t
 
=
 
e
n
c
o
d
e
r
.
t
r
a
n
s
f
o
r
m
(
X
_
t
e
s
t
)
```


```python
X
_
t
r
a
i
n
.
h
e
a
d
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
w
o
 
a
d
d
i
t
i
o
n
a
l
 
v
a
r
i
a
b
l
e
s
 
`
R
a
i
n
T
o
d
a
y
_
0
`
 
a
n
d
 
`
R
a
i
n
T
o
d
a
y
_
1
`
 
a
r
e
 
c
r
e
a
t
e
d
 
f
r
o
m
 
`
R
a
i
n
T
o
d
a
y
`
 
v
a
r
i
a
b
l
e
.




N
o
w
,
 
I
 
w
i
l
l
 
c
r
e
a
t
e
 
t
h
e
 
`
X
_
t
r
a
i
n
`
 
t
r
a
i
n
i
n
g
 
s
e
t
.



```python
X
_
t
r
a
i
n
 
=
 
p
d
.
c
o
n
c
a
t
(
[
X
_
t
r
a
i
n
[
n
u
m
e
r
i
c
a
l
]
,
 
X
_
t
r
a
i
n
[
[
'
R
a
i
n
T
o
d
a
y
_
0
'
,
 
'
R
a
i
n
T
o
d
a
y
_
1
'
]
]
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
r
a
i
n
.
L
o
c
a
t
i
o
n
)
,
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
r
a
i
n
.
W
i
n
d
G
u
s
t
D
i
r
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
r
a
i
n
.
W
i
n
d
D
i
r
9
a
m
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
r
a
i
n
.
W
i
n
d
D
i
r
3
p
m
)
]
,
 
a
x
i
s
=
1
)
```


```python
X
_
t
r
a
i
n
.
h
e
a
d
(
)
```

S
i
m
i
l
a
r
l
y
,
 
I
 
w
i
l
l
 
c
r
e
a
t
e
 
t
h
e
 
`
X
_
t
e
s
t
`
 
t
e
s
t
i
n
g
 
s
e
t
.



```python
X
_
t
e
s
t
 
=
 
p
d
.
c
o
n
c
a
t
(
[
X
_
t
e
s
t
[
n
u
m
e
r
i
c
a
l
]
,
 
X
_
t
e
s
t
[
[
'
R
a
i
n
T
o
d
a
y
_
0
'
,
 
'
R
a
i
n
T
o
d
a
y
_
1
'
]
]
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
e
s
t
.
L
o
c
a
t
i
o
n
)
,
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
e
s
t
.
W
i
n
d
G
u
s
t
D
i
r
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
e
s
t
.
W
i
n
d
D
i
r
9
a
m
)
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
d
.
g
e
t
_
d
u
m
m
i
e
s
(
X
_
t
e
s
t
.
W
i
n
d
D
i
r
3
p
m
)
]
,
 
a
x
i
s
=
1
)
```


```python
X
_
t
e
s
t
.
h
e
a
d
(
)
```

W
e
 
n
o
w
 
h
a
v
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
i
n
g
 
s
e
t
 
r
e
a
d
y
 
f
o
r
 
m
o
d
e
l
 
b
u
i
l
d
i
n
g
.
 
B
e
f
o
r
e
 
t
h
a
t
,
 
w
e
 
s
h
o
u
l
d
 
m
a
p
 
a
l
l
 
t
h
e
 
f
e
a
t
u
r
e
 
v
a
r
i
a
b
l
e
s
 
o
n
t
o
 
t
h
e
 
s
a
m
e
 
s
c
a
l
e
.
 
I
t
 
i
s
 
c
a
l
l
e
d
 
`
f
e
a
t
u
r
e
 
s
c
a
l
i
n
g
`
.
 
I
 
w
i
l
l
 
d
o
 
i
t
 
a
s
 
f
o
l
l
o
w
s
.


#
 
*
*
1
1
.
 
F
e
a
t
u
r
e
 
S
c
a
l
i
n
g
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
1
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
X
_
t
r
a
i
n
.
d
e
s
c
r
i
b
e
(
)
```


```python
c
o
l
s
 
=
 
X
_
t
r
a
i
n
.
c
o
l
u
m
n
s
```


```python
f
r
o
m
 
s
k
l
e
a
r
n
.
p
r
e
p
r
o
c
e
s
s
i
n
g
 
i
m
p
o
r
t
 
M
i
n
M
a
x
S
c
a
l
e
r


s
c
a
l
e
r
 
=
 
M
i
n
M
a
x
S
c
a
l
e
r
(
)


X
_
t
r
a
i
n
 
=
 
s
c
a
l
e
r
.
f
i
t
_
t
r
a
n
s
f
o
r
m
(
X
_
t
r
a
i
n
)


X
_
t
e
s
t
 
=
 
s
c
a
l
e
r
.
t
r
a
n
s
f
o
r
m
(
X
_
t
e
s
t
)

```


```python
X
_
t
r
a
i
n
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
X
_
t
r
a
i
n
,
 
c
o
l
u
m
n
s
=
[
c
o
l
s
]
)
```


```python
X
_
t
e
s
t
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
X
_
t
e
s
t
,
 
c
o
l
u
m
n
s
=
[
c
o
l
s
]
)
```


```python
X
_
t
r
a
i
n
.
d
e
s
c
r
i
b
e
(
)
```

W
e
 
n
o
w
 
h
a
v
e
 
`
X
_
t
r
a
i
n
`
 
d
a
t
a
s
e
t
 
r
e
a
d
y
 
t
o
 
b
e
 
f
e
d
 
i
n
t
o
 
t
h
e
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
c
l
a
s
s
i
f
i
e
r
.
 
I
 
w
i
l
l
 
d
o
 
i
t
 
a
s
 
f
o
l
l
o
w
s
.


#
 
*
*
1
2
.
 
M
o
d
e
l
 
t
r
a
i
n
i
n
g
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
2
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
#
 
t
r
a
i
n
 
a
 
l
o
g
i
s
t
i
c
 
r
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
o
n
 
t
h
e
 
t
r
a
i
n
i
n
g
 
s
e
t

f
r
o
m
 
s
k
l
e
a
r
n
.
l
i
n
e
a
r
_
m
o
d
e
l
 
i
m
p
o
r
t
 
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n



#
 
i
n
s
t
a
n
t
i
a
t
e
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
 
=
 
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n
(
s
o
l
v
e
r
=
'
l
i
b
l
i
n
e
a
r
'
,
 
r
a
n
d
o
m
_
s
t
a
t
e
=
0
)



#
 
f
i
t
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
.
f
i
t
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)

```

#
 
*
*
1
3
.
 
P
r
e
d
i
c
t
 
r
e
s
u
l
t
s
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
3
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
y
_
p
r
e
d
_
t
e
s
t
 
=
 
l
o
g
r
e
g
.
p
r
e
d
i
c
t
(
X
_
t
e
s
t
)


y
_
p
r
e
d
_
t
e
s
t
```

#
#
#
 
p
r
e
d
i
c
t
_
p
r
o
b
a
 
m
e
t
h
o
d






*
*
p
r
e
d
i
c
t
_
p
r
o
b
a
*
*
 
m
e
t
h
o
d
 
g
i
v
e
s
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
f
o
r
 
t
h
e
 
t
a
r
g
e
t
 
v
a
r
i
a
b
l
e
(
0
 
a
n
d
 
1
)
 
i
n
 
t
h
i
s
 
c
a
s
e
,
 
i
n
 
a
r
r
a
y
 
f
o
r
m
.




`
0
 
i
s
 
f
o
r
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
n
o
 
r
a
i
n
`
 
a
n
d
 
`
1
 
i
s
 
f
o
r
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
r
a
i
n
.
`



```python
#
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
g
e
t
t
i
n
g
 
o
u
t
p
u
t
 
a
s
 
0
 
-
 
n
o
 
r
a
i
n


l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
:
,
0
]
```


```python
#
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
g
e
t
t
i
n
g
 
o
u
t
p
u
t
 
a
s
 
1
 
-
 
r
a
i
n


l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
:
,
1
]
```

#
 
*
*
1
4
.
 
C
h
e
c
k
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
4
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
a
c
c
u
r
a
c
y
_
s
c
o
r
e


p
r
i
n
t
(
'
M
o
d
e
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
:
 
{
0
:
0
.
4
f
}
'
.
 
f
o
r
m
a
t
(
a
c
c
u
r
a
c
y
_
s
c
o
r
e
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
_
t
e
s
t
)
)
)
```

H
e
r
e
,
 
*
*
y
_
t
e
s
t
*
*
 
a
r
e
 
t
h
e
 
t
r
u
e
 
c
l
a
s
s
 
l
a
b
e
l
s
 
a
n
d
 
*
*
y
_
p
r
e
d
_
t
e
s
t
*
*
 
a
r
e
 
t
h
e
 
p
r
e
d
i
c
t
e
d
 
c
l
a
s
s
 
l
a
b
e
l
s
 
i
n
 
t
h
e
 
t
e
s
t
-
s
e
t
.


#
#
#
 
C
o
m
p
a
r
e
 
t
h
e
 
t
r
a
i
n
-
s
e
t
 
a
n
d
 
t
e
s
t
-
s
e
t
 
a
c
c
u
r
a
c
y






N
o
w
,
 
I
 
w
i
l
l
 
c
o
m
p
a
r
e
 
t
h
e
 
t
r
a
i
n
-
s
e
t
 
a
n
d
 
t
e
s
t
-
s
e
t
 
a
c
c
u
r
a
c
y
 
t
o
 
c
h
e
c
k
 
f
o
r
 
o
v
e
r
f
i
t
t
i
n
g
.



```python
y
_
p
r
e
d
_
t
r
a
i
n
 
=
 
l
o
g
r
e
g
.
p
r
e
d
i
c
t
(
X
_
t
r
a
i
n
)


y
_
p
r
e
d
_
t
r
a
i
n
```


```python
p
r
i
n
t
(
'
T
r
a
i
n
i
n
g
-
s
e
t
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
:
 
{
0
:
0
.
4
f
}
'
.
 
f
o
r
m
a
t
(
a
c
c
u
r
a
c
y
_
s
c
o
r
e
(
y
_
t
r
a
i
n
,
 
y
_
p
r
e
d
_
t
r
a
i
n
)
)
)
```

#
#
#
 
C
h
e
c
k
 
f
o
r
 
o
v
e
r
f
i
t
t
i
n
g
 
a
n
d
 
u
n
d
e
r
f
i
t
t
i
n
g



```python
#
 
p
r
i
n
t
 
t
h
e
 
s
c
o
r
e
s
 
o
n
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t


p
r
i
n
t
(
'
T
r
a
i
n
i
n
g
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
.
s
c
o
r
e
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)
)
)


p
r
i
n
t
(
'
T
e
s
t
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
.
s
c
o
r
e
(
X
_
t
e
s
t
,
 
y
_
t
e
s
t
)
)
)
```

T
h
e
 
t
r
a
i
n
i
n
g
-
s
e
t
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
i
s
 
0
.
8
4
7
6
 
w
h
i
l
e
 
t
h
e
 
t
e
s
t
-
s
e
t
 
a
c
c
u
r
a
c
y
 
t
o
 
b
e
 
0
.
8
5
0
1
.
 
T
h
e
s
e
 
t
w
o
 
v
a
l
u
e
s
 
a
r
e
 
q
u
i
t
e
 
c
o
m
p
a
r
a
b
l
e
.
 
S
o
,
 
t
h
e
r
e
 
i
s
 
n
o
 
q
u
e
s
t
i
o
n
 
o
f
 
o
v
e
r
f
i
t
t
i
n
g
.
 




I
n
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
,
 
w
e
 
u
s
e
 
d
e
f
a
u
l
t
 
v
a
l
u
e
 
o
f
 
C
 
=
 
1
.
 
I
t
 
p
r
o
v
i
d
e
s
 
g
o
o
d
 
p
e
r
f
o
r
m
a
n
c
e
 
w
i
t
h
 
a
p
p
r
o
x
i
m
a
t
e
l
y
 
8
5
%
 
a
c
c
u
r
a
c
y
 
o
n
 
b
o
t
h
 
t
h
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
h
e
 
t
e
s
t
 
s
e
t
.
 
B
u
t
 
t
h
e
 
m
o
d
e
l
 
p
e
r
f
o
r
m
a
n
c
e
 
o
n
 
b
o
t
h
 
t
h
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t
 
a
r
e
 
v
e
r
y
 
c
o
m
p
a
r
a
b
l
e
.
 
I
t
 
i
s
 
l
i
k
e
l
y
 
t
h
e
 
c
a
s
e
 
o
f
 
u
n
d
e
r
f
i
t
t
i
n
g
.
 




I
 
w
i
l
l
 
i
n
c
r
e
a
s
e
 
C
 
a
n
d
 
f
i
t
 
a
 
m
o
r
e
 
f
l
e
x
i
b
l
e
 
m
o
d
e
l
.



```python
#
 
f
i
t
 
t
h
e
 
L
o
g
s
i
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
w
i
t
h
 
C
=
1
0
0


#
 
i
n
s
t
a
n
t
i
a
t
e
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
1
0
0
 
=
 
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n
(
C
=
1
0
0
,
 
s
o
l
v
e
r
=
'
l
i
b
l
i
n
e
a
r
'
,
 
r
a
n
d
o
m
_
s
t
a
t
e
=
0
)



#
 
f
i
t
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
1
0
0
.
f
i
t
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)
```


```python
#
 
p
r
i
n
t
 
t
h
e
 
s
c
o
r
e
s
 
o
n
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t


p
r
i
n
t
(
'
T
r
a
i
n
i
n
g
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
1
0
0
.
s
c
o
r
e
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)
)
)


p
r
i
n
t
(
'
T
e
s
t
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
1
0
0
.
s
c
o
r
e
(
X
_
t
e
s
t
,
 
y
_
t
e
s
t
)
)
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
,
 
C
=
1
0
0
 
r
e
s
u
l
t
s
 
i
n
 
h
i
g
h
e
r
 
t
e
s
t
 
s
e
t
 
a
c
c
u
r
a
c
y
 
a
n
d
 
a
l
s
o
 
a
 
s
l
i
g
h
t
l
y
 
i
n
c
r
e
a
s
e
d
 
t
r
a
i
n
i
n
g
 
s
e
t
 
a
c
c
u
r
a
c
y
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
a
 
m
o
r
e
 
c
o
m
p
l
e
x
 
m
o
d
e
l
 
s
h
o
u
l
d
 
p
e
r
f
o
r
m
 
b
e
t
t
e
r
.


N
o
w
,
 
I
 
w
i
l
l
 
i
n
v
e
s
t
i
g
a
t
e
,
 
w
h
a
t
 
h
a
p
p
e
n
s
 
i
f
 
w
e
 
u
s
e
 
m
o
r
e
 
r
e
g
u
l
a
r
i
z
e
d
 
m
o
d
e
l
 
t
h
a
n
 
t
h
e
 
d
e
f
a
u
l
t
 
v
a
l
u
e
 
o
f
 
C
=
1
,
 
b
y
 
s
e
t
t
i
n
g
 
C
=
0
.
0
1
.



```python
#
 
f
i
t
 
t
h
e
 
L
o
g
s
i
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
w
i
t
h
 
C
=
0
0
1


#
 
i
n
s
t
a
n
t
i
a
t
e
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
0
0
1
 
=
 
L
o
g
i
s
t
i
c
R
e
g
r
e
s
s
i
o
n
(
C
=
0
.
0
1
,
 
s
o
l
v
e
r
=
'
l
i
b
l
i
n
e
a
r
'
,
 
r
a
n
d
o
m
_
s
t
a
t
e
=
0
)



#
 
f
i
t
 
t
h
e
 
m
o
d
e
l

l
o
g
r
e
g
0
0
1
.
f
i
t
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)
```


```python
#
 
p
r
i
n
t
 
t
h
e
 
s
c
o
r
e
s
 
o
n
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t


p
r
i
n
t
(
'
T
r
a
i
n
i
n
g
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
0
0
1
.
s
c
o
r
e
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)
)
)


p
r
i
n
t
(
'
T
e
s
t
 
s
e
t
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
l
o
g
r
e
g
0
0
1
.
s
c
o
r
e
(
X
_
t
e
s
t
,
 
y
_
t
e
s
t
)
)
)
```

S
o
,
 
i
f
 
w
e
 
u
s
e
 
m
o
r
e
 
r
e
g
u
l
a
r
i
z
e
d
 
m
o
d
e
l
 
b
y
 
s
e
t
t
i
n
g
 
C
=
0
.
0
1
,
 
t
h
e
n
 
b
o
t
h
 
t
h
e
 
t
r
a
i
n
i
n
g
 
a
n
d
 
t
e
s
t
 
s
e
t
 
a
c
c
u
r
a
c
y
 
d
e
c
r
e
a
s
e
 
r
e
l
a
t
i
e
v
 
t
o
 
t
h
e
 
d
e
f
a
u
l
t
 
p
a
r
a
m
e
t
e
r
s
.


#
#
#
 
C
o
m
p
a
r
e
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
w
i
t
h
 
n
u
l
l
 
a
c
c
u
r
a
c
y






S
o
,
 
t
h
e
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
i
s
 
0
.
8
5
0
1
.
 
B
u
t
,
 
w
e
 
c
a
n
n
o
t
 
s
a
y
 
t
h
a
t
 
o
u
r
 
m
o
d
e
l
 
i
s
 
v
e
r
y
 
g
o
o
d
 
b
a
s
e
d
 
o
n
 
t
h
e
 
a
b
o
v
e
 
a
c
c
u
r
a
c
y
.
 
W
e
 
m
u
s
t
 
c
o
m
p
a
r
e
 
i
t
 
w
i
t
h
 
t
h
e
 
*
*
n
u
l
l
 
a
c
c
u
r
a
c
y
*
*
.
 
N
u
l
l
 
a
c
c
u
r
a
c
y
 
i
s
 
t
h
e
 
a
c
c
u
r
a
c
y
 
t
h
a
t
 
c
o
u
l
d
 
b
e
 
a
c
h
i
e
v
e
d
 
b
y
 
a
l
w
a
y
s
 
p
r
e
d
i
c
t
i
n
g
 
t
h
e
 
m
o
s
t
 
f
r
e
q
u
e
n
t
 
c
l
a
s
s
.




S
o
,
 
w
e
 
s
h
o
u
l
d
 
f
i
r
s
t
 
c
h
e
c
k
 
t
h
e
 
c
l
a
s
s
 
d
i
s
t
r
i
b
u
t
i
o
n
 
i
n
 
t
h
e
 
t
e
s
t
 
s
e
t
.
 



```python
#
 
c
h
e
c
k
 
c
l
a
s
s
 
d
i
s
t
r
i
b
u
t
i
o
n
 
i
n
 
t
e
s
t
 
s
e
t


y
_
t
e
s
t
.
v
a
l
u
e
_
c
o
u
n
t
s
(
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
 
o
c
c
u
r
e
n
c
e
s
 
o
f
 
m
o
s
t
 
f
r
e
q
u
e
n
t
 
c
l
a
s
s
 
i
s
 
2
2
0
6
7
.
 
S
o
,
 
w
e
 
c
a
n
 
c
a
l
c
u
l
a
t
e
 
n
u
l
l
 
a
c
c
u
r
a
c
y
 
b
y
 
d
i
v
i
d
i
n
g
 
2
2
0
6
7
 
b
y
 
t
o
t
a
l
 
n
u
m
b
e
r
 
o
f
 
o
c
c
u
r
e
n
c
e
s
.



```python
#
 
c
h
e
c
k
 
n
u
l
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e


n
u
l
l
_
a
c
c
u
r
a
c
y
 
=
 
(
2
2
0
6
7
/
(
2
2
0
6
7
+
6
3
7
2
)
)


p
r
i
n
t
(
'
N
u
l
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
:
 
{
0
:
0
.
4
f
}
'
.
 
f
o
r
m
a
t
(
n
u
l
l
_
a
c
c
u
r
a
c
y
)
)
```

W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
o
u
r
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
i
s
 
0
.
8
5
0
1
 
b
u
t
 
n
u
l
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
i
s
 
0
.
7
7
5
9
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
o
u
r
 
L
o
g
i
s
t
i
c
 
R
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
i
s
 
d
o
i
n
g
 
a
 
v
e
r
y
 
g
o
o
d
 
j
o
b
 
i
n
 
p
r
e
d
i
c
t
i
n
g
 
t
h
e
 
c
l
a
s
s
 
l
a
b
e
l
s
.


N
o
w
,
 
b
a
s
e
d
 
o
n
 
t
h
e
 
a
b
o
v
e
 
a
n
a
l
y
s
i
s
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
o
u
r
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
i
s
 
v
e
r
y
 
g
o
o
d
.
 
O
u
r
 
m
o
d
e
l
 
i
s
 
d
o
i
n
g
 
a
 
v
e
r
y
 
g
o
o
d
 
j
o
b
 
i
n
 
t
e
r
m
s
 
o
f
 
p
r
e
d
i
c
t
i
n
g
 
t
h
e
 
c
l
a
s
s
 
l
a
b
e
l
s
.






B
u
t
,
 
i
t
 
d
o
e
s
 
n
o
t
 
g
i
v
e
 
t
h
e
 
u
n
d
e
r
l
y
i
n
g
 
d
i
s
t
r
i
b
u
t
i
o
n
 
o
f
 
v
a
l
u
e
s
.
 
A
l
s
o
,
 
i
t
 
d
o
e
s
 
n
o
t
 
t
e
l
l
 
a
n
y
t
h
i
n
g
 
a
b
o
u
t
 
t
h
e
 
t
y
p
e
 
o
f
 
e
r
r
o
r
s
 
o
u
r
 
c
l
a
s
s
i
f
e
r
 
i
s
 
m
a
k
i
n
g
.
 






W
e
 
h
a
v
e
 
a
n
o
t
h
e
r
 
t
o
o
l
 
c
a
l
l
e
d
 
`
C
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
`
 
t
h
a
t
 
c
o
m
e
s
 
t
o
 
o
u
r
 
r
e
s
c
u
e
.


#
 
*
*
1
5
.
 
C
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
5
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)






A
 
c
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
 
i
s
 
a
 
t
o
o
l
 
f
o
r
 
s
u
m
m
a
r
i
z
i
n
g
 
t
h
e
 
p
e
r
f
o
r
m
a
n
c
e
 
o
f
 
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
l
g
o
r
i
t
h
m
.
 
A
 
c
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
 
w
i
l
l
 
g
i
v
e
 
u
s
 
a
 
c
l
e
a
r
 
p
i
c
t
u
r
e
 
o
f
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
p
e
r
f
o
r
m
a
n
c
e
 
a
n
d
 
t
h
e
 
t
y
p
e
s
 
o
f
 
e
r
r
o
r
s
 
p
r
o
d
u
c
e
d
 
b
y
 
t
h
e
 
m
o
d
e
l
.
 
I
t
 
g
i
v
e
s
 
u
s
 
a
 
s
u
m
m
a
r
y
 
o
f
 
c
o
r
r
e
c
t
 
a
n
d
 
i
n
c
o
r
r
e
c
t
 
p
r
e
d
i
c
t
i
o
n
s
 
b
r
o
k
e
n
 
d
o
w
n
 
b
y
 
e
a
c
h
 
c
a
t
e
g
o
r
y
.
 
T
h
e
 
s
u
m
m
a
r
y
 
i
s
 
r
e
p
r
e
s
e
n
t
e
d
 
i
n
 
a
 
t
a
b
u
l
a
r
 
f
o
r
m
.






F
o
u
r
 
t
y
p
e
s
 
o
f
 
o
u
t
c
o
m
e
s
 
a
r
e
 
p
o
s
s
i
b
l
e
 
w
h
i
l
e
 
e
v
a
l
u
a
t
i
n
g
 
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
p
e
r
f
o
r
m
a
n
c
e
.
 
T
h
e
s
e
 
f
o
u
r
 
o
u
t
c
o
m
e
s
 
a
r
e
 
d
e
s
c
r
i
b
e
d
 
b
e
l
o
w
:
-






*
*
T
r
u
e
 
P
o
s
i
t
i
v
e
s
 
(
T
P
)
*
*
 
–
 
T
r
u
e
 
P
o
s
i
t
i
v
e
s
 
o
c
c
u
r
 
w
h
e
n
 
w
e
 
p
r
e
d
i
c
t
 
a
n
 
o
b
s
e
r
v
a
t
i
o
n
 
b
e
l
o
n
g
s
 
t
o
 
a
 
c
e
r
t
a
i
n
 
c
l
a
s
s
 
a
n
d
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
a
c
t
u
a
l
l
y
 
b
e
l
o
n
g
s
 
t
o
 
t
h
a
t
 
c
l
a
s
s
.






*
*
T
r
u
e
 
N
e
g
a
t
i
v
e
s
 
(
T
N
)
*
*
 
–
 
T
r
u
e
 
N
e
g
a
t
i
v
e
s
 
o
c
c
u
r
 
w
h
e
n
 
w
e
 
p
r
e
d
i
c
t
 
a
n
 
o
b
s
e
r
v
a
t
i
o
n
 
d
o
e
s
 
n
o
t
 
b
e
l
o
n
g
 
t
o
 
a
 
c
e
r
t
a
i
n
 
c
l
a
s
s
 
a
n
d
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
a
c
t
u
a
l
l
y
 
d
o
e
s
 
n
o
t
 
b
e
l
o
n
g
 
t
o
 
t
h
a
t
 
c
l
a
s
s
.






*
*
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
 
(
F
P
)
*
*
 
–
 
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
 
o
c
c
u
r
 
w
h
e
n
 
w
e
 
p
r
e
d
i
c
t
 
a
n
 
o
b
s
e
r
v
a
t
i
o
n
 
b
e
l
o
n
g
s
 
t
o
 
a
 
 
 
 
c
e
r
t
a
i
n
 
c
l
a
s
s
 
b
u
t
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
a
c
t
u
a
l
l
y
 
d
o
e
s
 
n
o
t
 
b
e
l
o
n
g
 
t
o
 
t
h
a
t
 
c
l
a
s
s
.
 
T
h
i
s
 
t
y
p
e
 
o
f
 
e
r
r
o
r
 
i
s
 
c
a
l
l
e
d
 
*
*
T
y
p
e
 
I
 
e
r
r
o
r
.
*
*








*
*
F
a
l
s
e
 
N
e
g
a
t
i
v
e
s
 
(
F
N
)
*
*
 
–
 
F
a
l
s
e
 
N
e
g
a
t
i
v
e
s
 
o
c
c
u
r
 
w
h
e
n
 
w
e
 
p
r
e
d
i
c
t
 
a
n
 
o
b
s
e
r
v
a
t
i
o
n
 
d
o
e
s
 
n
o
t
 
b
e
l
o
n
g
 
t
o
 
a
 
c
e
r
t
a
i
n
 
c
l
a
s
s
 
b
u
t
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
 
a
c
t
u
a
l
l
y
 
b
e
l
o
n
g
s
 
t
o
 
t
h
a
t
 
c
l
a
s
s
.
 
T
h
i
s
 
i
s
 
a
 
v
e
r
y
 
s
e
r
i
o
u
s
 
e
r
r
o
r
 
a
n
d
 
i
t
 
i
s
 
c
a
l
l
e
d
 
*
*
T
y
p
e
 
I
I
 
e
r
r
o
r
.
*
*








T
h
e
s
e
 
f
o
u
r
 
o
u
t
c
o
m
e
s
 
a
r
e
 
s
u
m
m
a
r
i
z
e
d
 
i
n
 
a
 
c
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
 
g
i
v
e
n
 
b
e
l
o
w
.





```python
#
 
P
r
i
n
t
 
t
h
e
 
C
o
n
f
u
s
i
o
n
 
M
a
t
r
i
x
 
a
n
d
 
s
l
i
c
e
 
i
t
 
i
n
t
o
 
f
o
u
r
 
p
i
e
c
e
s


f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
c
o
n
f
u
s
i
o
n
_
m
a
t
r
i
x


c
m
 
=
 
c
o
n
f
u
s
i
o
n
_
m
a
t
r
i
x
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
_
t
e
s
t
)


p
r
i
n
t
(
'
C
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
\
n
\
n
'
,
 
c
m
)


p
r
i
n
t
(
'
\
n
T
r
u
e
 
P
o
s
i
t
i
v
e
s
(
T
P
)
 
=
 
'
,
 
c
m
[
0
,
0
]
)


p
r
i
n
t
(
'
\
n
T
r
u
e
 
N
e
g
a
t
i
v
e
s
(
T
N
)
 
=
 
'
,
 
c
m
[
1
,
1
]
)


p
r
i
n
t
(
'
\
n
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
(
F
P
)
 
=
 
'
,
 
c
m
[
0
,
1
]
)


p
r
i
n
t
(
'
\
n
F
a
l
s
e
 
N
e
g
a
t
i
v
e
s
(
F
N
)
 
=
 
'
,
 
c
m
[
1
,
0
]
)
```

T
h
e
 
c
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
 
s
h
o
w
s
 
`
2
0
8
9
2
 
+
 
3
2
8
5
 
=
 
2
4
1
7
7
 
c
o
r
r
e
c
t
 
p
r
e
d
i
c
t
i
o
n
s
`
 
a
n
d
 
`
3
0
8
7
 
+
 
1
1
7
5
 
=
 
4
2
6
2
 
i
n
c
o
r
r
e
c
t
 
p
r
e
d
i
c
t
i
o
n
s
`
.






I
n
 
t
h
i
s
 
c
a
s
e
,
 
w
e
 
h
a
v
e






-
 
`
T
r
u
e
 
P
o
s
i
t
i
v
e
s
`
 
(
A
c
t
u
a
l
 
P
o
s
i
t
i
v
e
:
1
 
a
n
d
 
P
r
e
d
i
c
t
 
P
o
s
i
t
i
v
e
:
1
)
 
-
 
2
0
8
9
2






-
 
`
T
r
u
e
 
N
e
g
a
t
i
v
e
s
`
 
(
A
c
t
u
a
l
 
N
e
g
a
t
i
v
e
:
0
 
a
n
d
 
P
r
e
d
i
c
t
 
N
e
g
a
t
i
v
e
:
0
)
 
-
 
3
2
8
5






-
 
`
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
`
 
(
A
c
t
u
a
l
 
N
e
g
a
t
i
v
e
:
0
 
b
u
t
 
P
r
e
d
i
c
t
 
P
o
s
i
t
i
v
e
:
1
)
 
-
 
1
1
7
5
 
`
(
T
y
p
e
 
I
 
e
r
r
o
r
)
`






-
 
`
F
a
l
s
e
 
N
e
g
a
t
i
v
e
s
`
 
(
A
c
t
u
a
l
 
P
o
s
i
t
i
v
e
:
1
 
b
u
t
 
P
r
e
d
i
c
t
 
N
e
g
a
t
i
v
e
:
0
)
 
-
 
3
0
8
7
 
`
(
T
y
p
e
 
I
I
 
e
r
r
o
r
)
`



```python
#
 
v
i
s
u
a
l
i
z
e
 
c
o
n
f
u
s
i
o
n
 
m
a
t
r
i
x
 
w
i
t
h
 
s
e
a
b
o
r
n
 
h
e
a
t
m
a
p


c
m
_
m
a
t
r
i
x
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
d
a
t
a
=
c
m
,
 
c
o
l
u
m
n
s
=
[
'
A
c
t
u
a
l
 
P
o
s
i
t
i
v
e
:
1
'
,
 
'
A
c
t
u
a
l
 
N
e
g
a
t
i
v
e
:
0
'
]
,
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
i
n
d
e
x
=
[
'
P
r
e
d
i
c
t
 
P
o
s
i
t
i
v
e
:
1
'
,
 
'
P
r
e
d
i
c
t
 
N
e
g
a
t
i
v
e
:
0
'
]
)


s
n
s
.
h
e
a
t
m
a
p
(
c
m
_
m
a
t
r
i
x
,
 
a
n
n
o
t
=
T
r
u
e
,
 
f
m
t
=
'
d
'
,
 
c
m
a
p
=
'
Y
l
G
n
B
u
'
)
```

#
 
*
*
1
6
.
 
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
e
t
r
i
c
e
s
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
6
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)


#
#
 
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
R
e
p
o
r
t






*
*
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
r
e
p
o
r
t
*
*
 
i
s
 
a
n
o
t
h
e
r
 
w
a
y
 
t
o
 
e
v
a
l
u
a
t
e
 
t
h
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
p
e
r
f
o
r
m
a
n
c
e
.
 
I
t
 
d
i
s
p
l
a
y
s
 
t
h
e
 
 
*
*
p
r
e
c
i
s
i
o
n
*
*
,
 
*
*
r
e
c
a
l
l
*
*
,
 
*
*
f
1
*
*
 
a
n
d
 
*
*
s
u
p
p
o
r
t
*
*
 
s
c
o
r
e
s
 
f
o
r
 
t
h
e
 
m
o
d
e
l
.
 
I
 
h
a
v
e
 
d
e
s
c
r
i
b
e
d
 
t
h
e
s
e
 
t
e
r
m
s
 
i
n
 
l
a
t
e
r
.




W
e
 
c
a
n
 
p
r
i
n
t
 
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
r
e
p
o
r
t
 
a
s
 
f
o
l
l
o
w
s
:
-



```python
f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
r
e
p
o
r
t


p
r
i
n
t
(
c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
r
e
p
o
r
t
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
_
t
e
s
t
)
)
```

#
#
 
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
c
c
u
r
a
c
y



```python
T
P
 
=
 
c
m
[
0
,
0
]

T
N
 
=
 
c
m
[
1
,
1
]

F
P
 
=
 
c
m
[
0
,
1
]

F
N
 
=
 
c
m
[
1
,
0
]
```


```python
#
 
p
r
i
n
t
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
c
c
u
r
a
c
y


c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
a
c
c
u
r
a
c
y
 
=
 
(
T
P
 
+
 
T
N
)
 
/
 
f
l
o
a
t
(
T
P
 
+
 
T
N
 
+
 
F
P
 
+
 
F
N
)


p
r
i
n
t
(
'
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
a
c
c
u
r
a
c
y
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
a
c
c
u
r
a
c
y
)
)

```

#
#
 
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
e
r
r
o
r



```python
#
 
p
r
i
n
t
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
e
r
r
o
r


c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
e
r
r
o
r
 
=
 
(
F
P
 
+
 
F
N
)
 
/
 
f
l
o
a
t
(
T
P
 
+
 
T
N
 
+
 
F
P
 
+
 
F
N
)


p
r
i
n
t
(
'
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
e
r
r
o
r
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
c
l
a
s
s
i
f
i
c
a
t
i
o
n
_
e
r
r
o
r
)
)

```

#
#
 
P
r
e
c
i
s
i
o
n






*
*
P
r
e
c
i
s
i
o
n
*
*
 
c
a
n
 
b
e
 
d
e
f
i
n
e
d
 
a
s
 
t
h
e
 
p
e
r
c
e
n
t
a
g
e
 
o
f
 
c
o
r
r
e
c
t
l
y
 
p
r
e
d
i
c
t
e
d
 
p
o
s
i
t
i
v
e
 
o
u
t
c
o
m
e
s
 
o
u
t
 
o
f
 
a
l
l
 
t
h
e
 
p
r
e
d
i
c
t
e
d
 
p
o
s
i
t
i
v
e
 
o
u
t
c
o
m
e
s
.
 
I
t
 
c
a
n
 
b
e
 
g
i
v
e
n
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
t
r
u
e
 
p
o
s
i
t
i
v
e
s
 
(
T
P
)
 
t
o
 
t
h
e
 
s
u
m
 
o
f
 
t
r
u
e
 
a
n
d
 
f
a
l
s
e
 
p
o
s
i
t
i
v
e
s
 
(
T
P
 
+
 
F
P
)
.
 






S
o
,
 
*
*
P
r
e
c
i
s
i
o
n
*
*
 
i
d
e
n
t
i
f
i
e
s
 
t
h
e
 
p
r
o
p
o
r
t
i
o
n
 
o
f
 
c
o
r
r
e
c
t
l
y
 
p
r
e
d
i
c
t
e
d
 
p
o
s
i
t
i
v
e
 
o
u
t
c
o
m
e
.
 
I
t
 
i
s
 
m
o
r
e
 
c
o
n
c
e
r
n
e
d
 
w
i
t
h
 
t
h
e
 
p
o
s
i
t
i
v
e
 
c
l
a
s
s
 
t
h
a
n
 
t
h
e
 
n
e
g
a
t
i
v
e
 
c
l
a
s
s
.








M
a
t
h
e
m
a
t
i
c
a
l
l
y
,
 
p
r
e
c
i
s
i
o
n
 
c
a
n
 
b
e
 
d
e
f
i
n
e
d
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
`
T
P
 
t
o
 
(
T
P
 
+
 
F
P
)
.
`









```python
#
 
p
r
i
n
t
 
p
r
e
c
i
s
i
o
n
 
s
c
o
r
e


p
r
e
c
i
s
i
o
n
 
=
 
T
P
 
/
 
f
l
o
a
t
(
T
P
 
+
 
F
P
)



p
r
i
n
t
(
'
P
r
e
c
i
s
i
o
n
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
p
r
e
c
i
s
i
o
n
)
)

```

#
#
 
R
e
c
a
l
l






R
e
c
a
l
l
 
c
a
n
 
b
e
 
d
e
f
i
n
e
d
 
a
s
 
t
h
e
 
p
e
r
c
e
n
t
a
g
e
 
o
f
 
c
o
r
r
e
c
t
l
y
 
p
r
e
d
i
c
t
e
d
 
p
o
s
i
t
i
v
e
 
o
u
t
c
o
m
e
s
 
o
u
t
 
o
f
 
a
l
l
 
t
h
e
 
a
c
t
u
a
l
 
p
o
s
i
t
i
v
e
 
o
u
t
c
o
m
e
s
.


I
t
 
c
a
n
 
b
e
 
g
i
v
e
n
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
t
r
u
e
 
p
o
s
i
t
i
v
e
s
 
(
T
P
)
 
t
o
 
t
h
e
 
s
u
m
 
o
f
 
t
r
u
e
 
p
o
s
i
t
i
v
e
s
 
a
n
d
 
f
a
l
s
e
 
n
e
g
a
t
i
v
e
s
 
(
T
P
 
+
 
F
N
)
.
 
*
*
R
e
c
a
l
l
*
*
 
i
s
 
a
l
s
o
 
c
a
l
l
e
d
 
*
*
S
e
n
s
i
t
i
v
i
t
y
*
*
.






*
*
R
e
c
a
l
l
*
*
 
i
d
e
n
t
i
f
i
e
s
 
t
h
e
 
p
r
o
p
o
r
t
i
o
n
 
o
f
 
c
o
r
r
e
c
t
l
y
 
p
r
e
d
i
c
t
e
d
 
a
c
t
u
a
l
 
p
o
s
i
t
i
v
e
s
.






M
a
t
h
e
m
a
t
i
c
a
l
l
y
,
 
r
e
c
a
l
l
 
c
a
n
 
b
e
 
g
i
v
e
n
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
`
T
P
 
t
o
 
(
T
P
 
+
 
F
N
)
.
`











```python
r
e
c
a
l
l
 
=
 
T
P
 
/
 
f
l
o
a
t
(
T
P
 
+
 
F
N
)


p
r
i
n
t
(
'
R
e
c
a
l
l
 
o
r
 
S
e
n
s
i
t
i
v
i
t
y
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
r
e
c
a
l
l
)
)
```

#
#
 
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e






*
*
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
*
*
 
i
s
 
s
y
n
o
n
y
m
o
u
s
 
w
i
t
h
 
*
*
R
e
c
a
l
l
*
*
.





```python
t
r
u
e
_
p
o
s
i
t
i
v
e
_
r
a
t
e
 
=
 
T
P
 
/
 
f
l
o
a
t
(
T
P
 
+
 
F
N
)



p
r
i
n
t
(
'
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
t
r
u
e
_
p
o
s
i
t
i
v
e
_
r
a
t
e
)
)
```

#
#
 
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e



```python
f
a
l
s
e
_
p
o
s
i
t
i
v
e
_
r
a
t
e
 
=
 
F
P
 
/
 
f
l
o
a
t
(
F
P
 
+
 
T
N
)



p
r
i
n
t
(
'
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
f
a
l
s
e
_
p
o
s
i
t
i
v
e
_
r
a
t
e
)
)
```

#
#
 
S
p
e
c
i
f
i
c
i
t
y



```python
s
p
e
c
i
f
i
c
i
t
y
 
=
 
T
N
 
/
 
(
T
N
 
+
 
F
P
)


p
r
i
n
t
(
'
S
p
e
c
i
f
i
c
i
t
y
 
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
s
p
e
c
i
f
i
c
i
t
y
)
)
```

#
#
 
f
1
-
s
c
o
r
e






*
*
f
1
-
s
c
o
r
e
*
*
 
i
s
 
t
h
e
 
w
e
i
g
h
t
e
d
 
h
a
r
m
o
n
i
c
 
m
e
a
n
 
o
f
 
p
r
e
c
i
s
i
o
n
 
a
n
d
 
r
e
c
a
l
l
.
 
T
h
e
 
b
e
s
t
 
p
o
s
s
i
b
l
e
 
*
*
f
1
-
s
c
o
r
e
*
*
 
w
o
u
l
d
 
b
e
 
1
.
0
 
a
n
d
 
t
h
e
 
w
o
r
s
t
 


w
o
u
l
d
 
b
e
 
0
.
0
.
 
 
*
*
f
1
-
s
c
o
r
e
*
*
 
i
s
 
t
h
e
 
h
a
r
m
o
n
i
c
 
m
e
a
n
 
o
f
 
p
r
e
c
i
s
i
o
n
 
a
n
d
 
r
e
c
a
l
l
.
 
S
o
,
 
*
*
f
1
-
s
c
o
r
e
*
*
 
i
s
 
a
l
w
a
y
s
 
l
o
w
e
r
 
t
h
a
n
 
a
c
c
u
r
a
c
y
 
m
e
a
s
u
r
e
s
 
a
s
 
t
h
e
y
 
e
m
b
e
d
 
p
r
e
c
i
s
i
o
n
 
a
n
d
 
r
e
c
a
l
l
 
i
n
t
o
 
t
h
e
i
r
 
c
o
m
p
u
t
a
t
i
o
n
.
 
T
h
e
 
w
e
i
g
h
t
e
d
 
a
v
e
r
a
g
e
 
o
f
 
`
f
1
-
s
c
o
r
e
`
 
s
h
o
u
l
d
 
b
e
 
u
s
e
d
 
t
o
 


c
o
m
p
a
r
e
 
c
l
a
s
s
i
f
i
e
r
 
m
o
d
e
l
s
,
 
n
o
t
 
g
l
o
b
a
l
 
a
c
c
u
r
a
c
y
.






#
#
 
S
u
p
p
o
r
t






*
*
S
u
p
p
o
r
t
*
*
 
i
s
 
t
h
e
 
a
c
t
u
a
l
 
n
u
m
b
e
r
 
o
f
 
o
c
c
u
r
r
e
n
c
e
s
 
o
f
 
t
h
e
 
c
l
a
s
s
 
i
n
 
o
u
r
 
d
a
t
a
s
e
t
.


#
 
*
*
1
7
.
 
A
d
j
u
s
t
i
n
g
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
7
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
#
 
p
r
i
n
t
 
t
h
e
 
f
i
r
s
t
 
1
0
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
o
f
 
t
w
o
 
c
l
a
s
s
e
s
-
 
0
 
a
n
d
 
1


y
_
p
r
e
d
_
p
r
o
b
 
=
 
l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
0
:
1
0
]


y
_
p
r
e
d
_
p
r
o
b
```

#
#
#
 
O
b
s
e
r
v
a
t
i
o
n
s






-
 
I
n
 
e
a
c
h
 
r
o
w
,
 
t
h
e
 
n
u
m
b
e
r
s
 
s
u
m
 
t
o
 
1
.






-
 
T
h
e
r
e
 
a
r
e
 
2
 
c
o
l
u
m
n
s
 
w
h
i
c
h
 
c
o
r
r
e
s
p
o
n
d
 
t
o
 
2
 
c
l
a
s
s
e
s
 
-
 
0
 
a
n
d
 
1
.




 
 
 
 
-
 
C
l
a
s
s
 
0
 
-
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
y
 
t
h
a
t
 
t
h
e
r
e
 
i
s
 
n
o
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.
 
 
 
 


 
 
 
 


 
 
 
 
-
 
C
l
a
s
s
 
1
 
-
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
y
 
t
h
a
t
 
t
h
e
r
e
 
i
s
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.


 
 
 
 
 
 
 
 


 
 
 
 


-
 
I
m
p
o
r
t
a
n
c
e
 
o
f
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s




 
 
 
 
-
 
W
e
 
c
a
n
 
r
a
n
k
 
t
h
e
 
o
b
s
e
r
v
a
t
i
o
n
s
 
b
y
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
r
a
i
n
 
o
r
 
n
o
 
r
a
i
n
.






-
 
p
r
e
d
i
c
t
_
p
r
o
b
a
 
p
r
o
c
e
s
s




 
 
 
 
-
 
P
r
e
d
i
c
t
s
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
 
 
 


 
 
 
 


 
 
 
 
-
 
C
h
o
o
s
e
 
t
h
e
 
c
l
a
s
s
 
w
i
t
h
 
t
h
e
 
h
i
g
h
e
s
t
 
p
r
o
b
a
b
i
l
i
t
y
 
 
 
 


 
 
 
 


 
 
 
 


-
 
C
l
a
s
s
i
f
i
c
a
t
i
o
n
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l




 
 
 
 
-
 
T
h
e
r
e
 
i
s
 
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
 
o
f
 
0
.
5
.
 
 
 
 


 
 
 
 


 
 
 
 
-
 
C
l
a
s
s
 
1
 
-
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
r
a
i
n
 
i
s
 
p
r
e
d
i
c
t
e
d
 
i
f
 
p
r
o
b
a
b
i
l
i
t
y
 
>
 
0
.
5
.
 
 
 
 


 
 
 
 


 
 
 
 
-
 
C
l
a
s
s
 
0
 
-
 
p
r
o
b
a
b
i
l
i
t
y
 
o
f
 
n
o
 
r
a
i
n
 
i
s
 
p
r
e
d
i
c
t
e
d
 
i
f
 
p
r
o
b
a
b
i
l
i
t
y
 
<
 
0
.
5
.
 
 
 
 


 
 
 
 





```python
#
 
s
t
o
r
e
 
t
h
e
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
i
n
 
d
a
t
a
f
r
a
m
e


y
_
p
r
e
d
_
p
r
o
b
_
d
f
 
=
 
p
d
.
D
a
t
a
F
r
a
m
e
(
d
a
t
a
=
y
_
p
r
e
d
_
p
r
o
b
,
 
c
o
l
u
m
n
s
=
[
'
P
r
o
b
 
o
f
 
-
 
N
o
 
r
a
i
n
 
t
o
m
o
r
r
o
w
 
(
0
)
'
,
 
'
P
r
o
b
 
o
f
 
-
 
R
a
i
n
 
t
o
m
o
r
r
o
w
 
(
1
)
'
]
)


y
_
p
r
e
d
_
p
r
o
b
_
d
f
```


```python
#
 
p
r
i
n
t
 
t
h
e
 
f
i
r
s
t
 
1
0
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
f
o
r
 
c
l
a
s
s
 
1
 
-
 
P
r
o
b
a
b
i
l
i
t
y
 
o
f
 
r
a
i
n


l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
0
:
1
0
,
 
1
]
```


```python
#
 
s
t
o
r
e
 
t
h
e
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
f
o
r
 
c
l
a
s
s
 
1
 
-
 
P
r
o
b
a
b
i
l
i
t
y
 
o
f
 
r
a
i
n


y
_
p
r
e
d
1
 
=
 
l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
:
,
 
1
]
```


```python
#
 
p
l
o
t
 
h
i
s
t
o
g
r
a
m
 
o
f
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s



#
 
a
d
j
u
s
t
 
t
h
e
 
f
o
n
t
 
s
i
z
e
 

p
l
t
.
r
c
P
a
r
a
m
s
[
'
f
o
n
t
.
s
i
z
e
'
]
 
=
 
1
2



#
 
p
l
o
t
 
h
i
s
t
o
g
r
a
m
 
w
i
t
h
 
1
0
 
b
i
n
s

p
l
t
.
h
i
s
t
(
y
_
p
r
e
d
1
,
 
b
i
n
s
 
=
 
1
0
)



#
 
s
e
t
 
t
h
e
 
t
i
t
l
e
 
o
f
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s

p
l
t
.
t
i
t
l
e
(
'
H
i
s
t
o
g
r
a
m
 
o
f
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
o
f
 
r
a
i
n
'
)



#
 
s
e
t
 
t
h
e
 
x
-
a
x
i
s
 
l
i
m
i
t

p
l
t
.
x
l
i
m
(
0
,
1
)



#
 
s
e
t
 
t
h
e
 
t
i
t
l
e

p
l
t
.
x
l
a
b
e
l
(
'
P
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
o
f
 
r
a
i
n
'
)

p
l
t
.
y
l
a
b
e
l
(
'
F
r
e
q
u
e
n
c
y
'
)
```

#
#
#
 
O
b
s
e
r
v
a
t
i
o
n
s






-
 
W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
t
h
e
 
a
b
o
v
e
 
h
i
s
t
o
g
r
a
m
 
i
s
 
h
i
g
h
l
y
 
p
o
s
i
t
i
v
e
 
s
k
e
w
e
d
.






-
 
T
h
e
 
f
i
r
s
t
 
c
o
l
u
m
n
 
t
e
l
l
 
u
s
 
t
h
a
t
 
t
h
e
r
e
 
a
r
e
 
a
p
p
r
o
x
i
m
a
t
e
l
y
 
1
5
0
0
0
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
i
t
h
 
p
r
o
b
a
b
i
l
i
t
y
 
b
e
t
w
e
e
n
 
0
.
0
 
a
n
d
 
0
.
1
.






-
 
T
h
e
r
e
 
a
r
e
 
s
m
a
l
l
 
n
u
m
b
e
r
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
 
w
i
t
h
 
p
r
o
b
a
b
i
l
i
t
y
 
>
 
0
.
5
.






-
 
S
o
,
 
t
h
e
s
e
 
s
m
a
l
l
 
n
u
m
b
e
r
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
 
p
r
e
d
i
c
t
 
t
h
a
t
 
t
h
e
r
e
 
w
i
l
l
 
b
e
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.






-
 
M
a
j
o
r
i
t
y
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
 
p
r
e
d
i
c
t
 
t
h
a
t
 
t
h
e
r
e
 
w
i
l
l
 
b
e
 
n
o
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.


#
#
#
 
L
o
w
e
r
 
t
h
e
 
t
h
r
e
s
h
o
l
d



```python
f
r
o
m
 
s
k
l
e
a
r
n
.
p
r
e
p
r
o
c
e
s
s
i
n
g
 
i
m
p
o
r
t
 
b
i
n
a
r
i
z
e


f
o
r
 
i
 
i
n
 
r
a
n
g
e
(
1
,
5
)
:

 
 
 
 

 
 
 
 
c
m
1
=
0

 
 
 
 

 
 
 
 
y
_
p
r
e
d
1
 
=
 
l
o
g
r
e
g
.
p
r
e
d
i
c
t
_
p
r
o
b
a
(
X
_
t
e
s
t
)
[
:
,
1
]

 
 
 
 

 
 
 
 
y
_
p
r
e
d
1
 
=
 
y
_
p
r
e
d
1
.
r
e
s
h
a
p
e
(
-
1
,
1
)

 
 
 
 

 
 
 
 
y
_
p
r
e
d
2
 
=
 
b
i
n
a
r
i
z
e
(
y
_
p
r
e
d
1
,
 
i
/
1
0
)

 
 
 
 

 
 
 
 
y
_
p
r
e
d
2
 
=
 
n
p
.
w
h
e
r
e
(
y
_
p
r
e
d
2
 
=
=
 
1
,
 
'
Y
e
s
'
,
 
'
N
o
'
)

 
 
 
 

 
 
 
 
c
m
1
 
=
 
c
o
n
f
u
s
i
o
n
_
m
a
t
r
i
x
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
2
)

 
 
 
 
 
 
 
 

 
 
 
 
p
r
i
n
t
 
(
'
W
i
t
h
'
,
i
/
1
0
,
'
t
h
r
e
s
h
o
l
d
 
t
h
e
 
C
o
n
f
u
s
i
o
n
 
M
a
t
r
i
x
 
i
s
 
'
,
'
\
n
\
n
'
,
c
m
1
,
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
'
w
i
t
h
'
,
c
m
1
[
0
,
0
]
+
c
m
1
[
1
,
1
]
,
'
c
o
r
r
e
c
t
 
p
r
e
d
i
c
t
i
o
n
s
,
 
'
,
 
'
\
n
\
n
'
,
 

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
c
m
1
[
0
,
1
]
,
'
T
y
p
e
 
I
 
e
r
r
o
r
s
(
 
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
)
,
 
'
,
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
c
m
1
[
1
,
0
]
,
'
T
y
p
e
 
I
I
 
e
r
r
o
r
s
(
 
F
a
l
s
e
 
N
e
g
a
t
i
v
e
s
)
,
 
'
,
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
'
A
c
c
u
r
a
c
y
 
s
c
o
r
e
:
 
'
,
 
(
a
c
c
u
r
a
c
y
_
s
c
o
r
e
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
2
)
)
,
 
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
'
S
e
n
s
i
t
i
v
i
t
y
:
 
'
,
c
m
1
[
1
,
1
]
/
(
f
l
o
a
t
(
c
m
1
[
1
,
1
]
+
c
m
1
[
1
,
0
]
)
)
,
 
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
'
S
p
e
c
i
f
i
c
i
t
y
:
 
'
,
c
m
1
[
0
,
0
]
/
(
f
l
o
a
t
(
c
m
1
[
0
,
0
]
+
c
m
1
[
0
,
1
]
)
)
,
'
\
n
\
n
'
,

 
 
 
 
 
 
 
 
 
 

 
 
 
 
 
 
 
 
 
 
 
 
'
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
=
'
,
 
'
\
n
\
n
'
)
```

#
#
#
 
C
o
m
m
e
n
t
s






-
 
I
n
 
b
i
n
a
r
y
 
p
r
o
b
l
e
m
s
,
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
o
f
 
0
.
5
 
i
s
 
u
s
e
d
 
b
y
 
d
e
f
a
u
l
t
 
t
o
 
c
o
n
v
e
r
t
 
p
r
e
d
i
c
t
e
d
 
p
r
o
b
a
b
i
l
i
t
i
e
s
 
i
n
t
o
 
c
l
a
s
s
 
p
r
e
d
i
c
t
i
o
n
s
.






-
 
T
h
r
e
s
h
o
l
d
 
c
a
n
 
b
e
 
a
d
j
u
s
t
e
d
 
t
o
 
i
n
c
r
e
a
s
e
 
s
e
n
s
i
t
i
v
i
t
y
 
o
r
 
s
p
e
c
i
f
i
c
i
t
y
.
 






-
 
S
e
n
s
i
t
i
v
i
t
y
 
a
n
d
 
s
p
e
c
i
f
i
c
i
t
y
 
h
a
v
e
 
a
n
 
i
n
v
e
r
s
e
 
r
e
l
a
t
i
o
n
s
h
i
p
.
 
I
n
c
r
e
a
s
i
n
g
 
o
n
e
 
w
o
u
l
d
 
a
l
w
a
y
s
 
d
e
c
r
e
a
s
e
 
t
h
e
 
o
t
h
e
r
 
a
n
d
 
v
i
c
e
 
v
e
r
s
a
.






-
 
W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
i
n
c
r
e
a
s
i
n
g
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
 
r
e
s
u
l
t
s
 
i
n
 
i
n
c
r
e
a
s
e
d
 
a
c
c
u
r
a
c
y
.






-
 
A
d
j
u
s
t
i
n
g
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
 
s
h
o
u
l
d
 
b
e
 
o
n
e
 
o
f
 
t
h
e
 
l
a
s
t
 
s
t
e
p
 
y
o
u
 
d
o
 
i
n
 
t
h
e
 
m
o
d
e
l
-
b
u
i
l
d
i
n
g
 
p
r
o
c
e
s
s
.


#
 
*
*
1
8
.
 
R
O
C
 
-
 
A
U
C
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
8
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)








#
#
 
R
O
C
 
C
u
r
v
e






A
n
o
t
h
e
r
 
t
o
o
l
 
t
o
 
m
e
a
s
u
r
e
 
t
h
e
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
p
e
r
f
o
r
m
a
n
c
e
 
v
i
s
u
a
l
l
y
 
i
s
 
*
*
R
O
C
 
C
u
r
v
e
*
*
.
 
R
O
C
 
C
u
r
v
e
 
s
t
a
n
d
s
 
f
o
r
 
*
*
R
e
c
e
i
v
e
r
 
O
p
e
r
a
t
i
n
g
 
C
h
a
r
a
c
t
e
r
i
s
t
i
c
 
C
u
r
v
e
*
*
.
 
A
n
 
*
*
R
O
C
 
C
u
r
v
e
*
*
 
i
s
 
a
 
p
l
o
t
 
w
h
i
c
h
 
s
h
o
w
s
 
t
h
e
 
p
e
r
f
o
r
m
a
n
c
e
 
o
f
 
a
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
m
o
d
e
l
 
a
t
 
v
a
r
i
o
u
s
 


c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
s
.
 








T
h
e
 
*
*
R
O
C
 
C
u
r
v
e
*
*
 
p
l
o
t
s
 
t
h
e
 
*
*
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
T
P
R
)
*
*
 
a
g
a
i
n
s
t
 
t
h
e
 
*
*
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
F
P
R
)
*
*
 
a
t
 
v
a
r
i
o
u
s
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
s
.








*
*
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
T
P
R
)
*
*
 
i
s
 
a
l
s
o
 
c
a
l
l
e
d
 
*
*
R
e
c
a
l
l
*
*
.
 
I
t
 
i
s
 
d
e
f
i
n
e
d
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
`
T
P
 
t
o
 
(
T
P
 
+
 
F
N
)
.
`








*
*
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
F
P
R
)
*
*
 
i
s
 
d
e
f
i
n
e
d
 
a
s
 
t
h
e
 
r
a
t
i
o
 
o
f
 
`
F
P
 
t
o
 
(
F
P
 
+
 
T
N
)
.
`










I
n
 
t
h
e
 
R
O
C
 
C
u
r
v
e
,
 
w
e
 
w
i
l
l
 
f
o
c
u
s
 
o
n
 
t
h
e
 
T
P
R
 
(
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
)
 
a
n
d
 
F
P
R
 
(
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
)
 
o
f
 
a
 
s
i
n
g
l
e
 
p
o
i
n
t
.
 
T
h
i
s
 
w
i
l
l
 
g
i
v
e
 
u
s
 
t
h
e
 
g
e
n
e
r
a
l
 
p
e
r
f
o
r
m
a
n
c
e
 
o
f
 
t
h
e
 
R
O
C
 
c
u
r
v
e
 
w
h
i
c
h
 
c
o
n
s
i
s
t
s
 
o
f
 
t
h
e
 
T
P
R
 
a
n
d
 
F
P
R
 
a
t
 
v
a
r
i
o
u
s
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
s
.
 
S
o
,
 
a
n
 
R
O
C
 
C
u
r
v
e
 
p
l
o
t
s
 
T
P
R
 
v
s
 
F
P
R
 
a
t
 
d
i
f
f
e
r
e
n
t
 
c
l
a
s
s
i
f
i
c
a
t
i
o
n
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
s
.
 
I
f
 
w
e
 
l
o
w
e
r
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
s
,
 
i
t
 
m
a
y
 
r
e
s
u
l
t
 
i
n
 
m
o
r
e
 
i
t
e
m
s
 
b
e
i
n
g
 
c
l
a
s
s
i
f
i
e
d
 
a
s
 
p
o
s
i
t
v
e
.
 
I
t
 
w
i
l
l
 
i
n
c
r
e
a
s
e
 
b
o
t
h
 
T
r
u
e
 
P
o
s
i
t
i
v
e
s
 
(
T
P
)
 
a
n
d
 
F
a
l
s
e
 
P
o
s
i
t
i
v
e
s
 
(
F
P
)
.







```python
#
 
p
l
o
t
 
R
O
C
 
C
u
r
v
e


f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
r
o
c
_
c
u
r
v
e


f
p
r
,
 
t
p
r
,
 
t
h
r
e
s
h
o
l
d
s
 
=
 
r
o
c
_
c
u
r
v
e
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
1
,
 
p
o
s
_
l
a
b
e
l
 
=
 
'
Y
e
s
'
)


p
l
t
.
f
i
g
u
r
e
(
f
i
g
s
i
z
e
=
(
6
,
4
)
)


p
l
t
.
p
l
o
t
(
f
p
r
,
 
t
p
r
,
 
l
i
n
e
w
i
d
t
h
=
2
)


p
l
t
.
p
l
o
t
(
[
0
,
1
]
,
 
[
0
,
1
]
,
 
'
k
-
-
'
 
)


p
l
t
.
r
c
P
a
r
a
m
s
[
'
f
o
n
t
.
s
i
z
e
'
]
 
=
 
1
2


p
l
t
.
t
i
t
l
e
(
'
R
O
C
 
c
u
r
v
e
 
f
o
r
 
R
a
i
n
T
o
m
o
r
r
o
w
 
c
l
a
s
s
i
f
i
e
r
'
)


p
l
t
.
x
l
a
b
e
l
(
'
F
a
l
s
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
1
 
-
 
S
p
e
c
i
f
i
c
i
t
y
)
'
)


p
l
t
.
y
l
a
b
e
l
(
'
T
r
u
e
 
P
o
s
i
t
i
v
e
 
R
a
t
e
 
(
S
e
n
s
i
t
i
v
i
t
y
)
'
)


p
l
t
.
s
h
o
w
(
)

```

R
O
C
 
c
u
r
v
e
 
h
e
l
p
 
u
s
 
t
o
 
c
h
o
o
s
e
 
a
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
 
t
h
a
t
 
b
a
l
a
n
c
e
s
 
s
e
n
s
i
t
i
v
i
t
y
 
a
n
d
 
s
p
e
c
i
f
i
c
i
t
y
 
f
o
r
 
a
 
p
a
r
t
i
c
u
l
a
r
 
c
o
n
t
e
x
t
.


#
#
 
R
O
C
-
A
U
C






*
*
R
O
C
 
A
U
C
*
*
 
s
t
a
n
d
s
 
f
o
r
 
*
*
R
e
c
e
i
v
e
r
 
O
p
e
r
a
t
i
n
g
 
C
h
a
r
a
c
t
e
r
i
s
t
i
c
 
-
 
A
r
e
a
 
U
n
d
e
r
 
C
u
r
v
e
*
*
.
 
I
t
 
i
s
 
a
 
t
e
c
h
n
i
q
u
e
 
t
o
 
c
o
m
p
a
r
e
 
c
l
a
s
s
i
f
i
e
r
 
p
e
r
f
o
r
m
a
n
c
e
.
 
I
n
 
t
h
i
s
 
t
e
c
h
n
i
q
u
e
,
 
w
e
 
m
e
a
s
u
r
e
 
t
h
e
 
`
a
r
e
a
 
u
n
d
e
r
 
t
h
e
 
c
u
r
v
e
 
(
A
U
C
)
`
.
 
A
 
p
e
r
f
e
c
t
 
c
l
a
s
s
i
f
i
e
r
 
w
i
l
l
 
h
a
v
e
 
a
 
R
O
C
 
A
U
C
 
e
q
u
a
l
 
t
o
 
1
,
 
w
h
e
r
e
a
s
 
a
 
p
u
r
e
l
y
 
r
a
n
d
o
m
 
c
l
a
s
s
i
f
i
e
r
 
w
i
l
l
 
h
a
v
e
 
a
 
R
O
C
 
A
U
C
 
e
q
u
a
l
 
t
o
 
0
.
5
.
 






S
o
,
 
*
*
R
O
C
 
A
U
C
*
*
 
i
s
 
t
h
e
 
p
e
r
c
e
n
t
a
g
e
 
o
f
 
t
h
e
 
R
O
C
 
p
l
o
t
 
t
h
a
t
 
i
s
 
u
n
d
e
r
n
e
a
t
h
 
t
h
e
 
c
u
r
v
e
.



```python
#
 
c
o
m
p
u
t
e
 
R
O
C
 
A
U
C


f
r
o
m
 
s
k
l
e
a
r
n
.
m
e
t
r
i
c
s
 
i
m
p
o
r
t
 
r
o
c
_
a
u
c
_
s
c
o
r
e


R
O
C
_
A
U
C
 
=
 
r
o
c
_
a
u
c
_
s
c
o
r
e
(
y
_
t
e
s
t
,
 
y
_
p
r
e
d
1
)


p
r
i
n
t
(
'
R
O
C
 
A
U
C
 
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
R
O
C
_
A
U
C
)
)
```

#
#
#
 
C
o
m
m
e
n
t
s






-
 
R
O
C
 
A
U
C
 
i
s
 
a
 
s
i
n
g
l
e
 
n
u
m
b
e
r
 
s
u
m
m
a
r
y
 
o
f
 
c
l
a
s
s
i
f
i
e
r
 
p
e
r
f
o
r
m
a
n
c
e
.
 
T
h
e
 
h
i
g
h
e
r
 
t
h
e
 
v
a
l
u
e
,
 
t
h
e
 
b
e
t
t
e
r
 
t
h
e
 
c
l
a
s
s
i
f
i
e
r
.




-
 
R
O
C
 
A
U
C
 
o
f
 
o
u
r
 
m
o
d
e
l
 
a
p
p
r
o
a
c
h
e
s
 
t
o
w
a
r
d
s
 
1
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
o
u
r
 
c
l
a
s
s
i
f
i
e
r
 
d
o
e
s
 
a
 
g
o
o
d
 
j
o
b
 
i
n
 
p
r
e
d
i
c
t
i
n
g
 
w
h
e
t
h
e
r
 
i
t
 
w
i
l
l
 
r
a
i
n
 
t
o
m
o
r
r
o
w
 
o
r
 
n
o
t
.



```python
#
 
c
a
l
c
u
l
a
t
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
e
d
 
R
O
C
 
A
U
C
 


f
r
o
m
 
s
k
l
e
a
r
n
.
m
o
d
e
l
_
s
e
l
e
c
t
i
o
n
 
i
m
p
o
r
t
 
c
r
o
s
s
_
v
a
l
_
s
c
o
r
e


C
r
o
s
s
_
v
a
l
i
d
a
t
e
d
_
R
O
C
_
A
U
C
 
=
 
c
r
o
s
s
_
v
a
l
_
s
c
o
r
e
(
l
o
g
r
e
g
,
 
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
,
 
c
v
=
5
,
 
s
c
o
r
i
n
g
=
'
r
o
c
_
a
u
c
'
)
.
m
e
a
n
(
)


p
r
i
n
t
(
'
C
r
o
s
s
 
v
a
l
i
d
a
t
e
d
 
R
O
C
 
A
U
C
 
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
C
r
o
s
s
_
v
a
l
i
d
a
t
e
d
_
R
O
C
_
A
U
C
)
)
```

#
 
*
*
1
9
.
 
k
-
F
o
l
d
 
C
r
o
s
s
 
V
a
l
i
d
a
t
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
1
9
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
#
 
A
p
p
l
y
i
n
g
 
5
-
F
o
l
d
 
C
r
o
s
s
 
V
a
l
i
d
a
t
i
o
n


f
r
o
m
 
s
k
l
e
a
r
n
.
m
o
d
e
l
_
s
e
l
e
c
t
i
o
n
 
i
m
p
o
r
t
 
c
r
o
s
s
_
v
a
l
_
s
c
o
r
e


s
c
o
r
e
s
 
=
 
c
r
o
s
s
_
v
a
l
_
s
c
o
r
e
(
l
o
g
r
e
g
,
 
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
,
 
c
v
 
=
 
5
,
 
s
c
o
r
i
n
g
=
'
a
c
c
u
r
a
c
y
'
)


p
r
i
n
t
(
'
C
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
s
c
o
r
e
s
:
{
}
'
.
f
o
r
m
a
t
(
s
c
o
r
e
s
)
)
```

W
e
 
c
a
n
 
s
u
m
m
a
r
i
z
e
 
t
h
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
a
c
c
u
r
a
c
y
 
b
y
 
c
a
l
c
u
l
a
t
i
n
g
 
i
t
s
 
m
e
a
n
.



```python
#
 
c
o
m
p
u
t
e
 
A
v
e
r
a
g
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
s
c
o
r
e


p
r
i
n
t
(
'
A
v
e
r
a
g
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
s
c
o
r
e
:
 
{
:
.
4
f
}
'
.
f
o
r
m
a
t
(
s
c
o
r
e
s
.
m
e
a
n
(
)
)
)
```

O
u
r
,
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
 
s
c
o
r
e
 
i
s
 
f
o
u
n
d
 
t
o
 
b
e
 
0
.
8
4
7
6
.
 
T
h
e
 
a
v
e
r
a
g
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
s
c
o
r
e
 
i
s
 
0
.
8
4
7
4
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
d
o
e
s
 
n
o
t
 
r
e
s
u
l
t
 
i
n
 
p
e
r
f
o
r
m
a
n
c
e
 
i
m
p
r
o
v
e
m
e
n
t
.


#
 
*
*
2
0
.
 
H
y
p
e
r
p
a
r
a
m
e
t
e
r
 
O
p
t
i
m
i
z
a
t
i
o
n
 
u
s
i
n
g
 
G
r
i
d
S
e
a
r
c
h
 
C
V
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
2
0
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)



```python
f
r
o
m
 
s
k
l
e
a
r
n
.
m
o
d
e
l
_
s
e
l
e
c
t
i
o
n
 
i
m
p
o
r
t
 
G
r
i
d
S
e
a
r
c
h
C
V



p
a
r
a
m
e
t
e
r
s
 
=
 
[
{
'
p
e
n
a
l
t
y
'
:
[
'
l
1
'
,
'
l
2
'
]
}
,
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
{
'
C
'
:
[
1
,
 
1
0
,
 
1
0
0
,
 
1
0
0
0
]
}
]




g
r
i
d
_
s
e
a
r
c
h
 
=
 
G
r
i
d
S
e
a
r
c
h
C
V
(
e
s
t
i
m
a
t
o
r
 
=
 
l
o
g
r
e
g
,
 
 

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
p
a
r
a
m
_
g
r
i
d
 
=
 
p
a
r
a
m
e
t
e
r
s
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
s
c
o
r
i
n
g
 
=
 
'
a
c
c
u
r
a
c
y
'
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
c
v
 
=
 
5
,

 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
v
e
r
b
o
s
e
=
0
)



g
r
i
d
_
s
e
a
r
c
h
.
f
i
t
(
X
_
t
r
a
i
n
,
 
y
_
t
r
a
i
n
)

```


```python
#
 
e
x
a
m
i
n
e
 
t
h
e
 
b
e
s
t
 
m
o
d
e
l


#
 
b
e
s
t
 
s
c
o
r
e
 
a
c
h
i
e
v
e
d
 
d
u
r
i
n
g
 
t
h
e
 
G
r
i
d
S
e
a
r
c
h
C
V

p
r
i
n
t
(
'
G
r
i
d
S
e
a
r
c
h
 
C
V
 
b
e
s
t
 
s
c
o
r
e
 
:
 
{
:
.
4
f
}
\
n
\
n
'
.
f
o
r
m
a
t
(
g
r
i
d
_
s
e
a
r
c
h
.
b
e
s
t
_
s
c
o
r
e
_
)
)


#
 
p
r
i
n
t
 
p
a
r
a
m
e
t
e
r
s
 
t
h
a
t
 
g
i
v
e
 
t
h
e
 
b
e
s
t
 
r
e
s
u
l
t
s

p
r
i
n
t
(
'
P
a
r
a
m
e
t
e
r
s
 
t
h
a
t
 
g
i
v
e
 
t
h
e
 
b
e
s
t
 
r
e
s
u
l
t
s
 
:
'
,
'
\
n
\
n
'
,
 
(
g
r
i
d
_
s
e
a
r
c
h
.
b
e
s
t
_
p
a
r
a
m
s
_
)
)


#
 
p
r
i
n
t
 
e
s
t
i
m
a
t
o
r
 
t
h
a
t
 
w
a
s
 
c
h
o
s
e
n
 
b
y
 
t
h
e
 
G
r
i
d
S
e
a
r
c
h

p
r
i
n
t
(
'
\
n
\
n
E
s
t
i
m
a
t
o
r
 
t
h
a
t
 
w
a
s
 
c
h
o
s
e
n
 
b
y
 
t
h
e
 
s
e
a
r
c
h
 
:
'
,
'
\
n
\
n
'
,
 
(
g
r
i
d
_
s
e
a
r
c
h
.
b
e
s
t
_
e
s
t
i
m
a
t
o
r
_
)
)
```


```python
#
 
c
a
l
c
u
l
a
t
e
 
G
r
i
d
S
e
a
r
c
h
 
C
V
 
s
c
o
r
e
 
o
n
 
t
e
s
t
 
s
e
t


p
r
i
n
t
(
'
G
r
i
d
S
e
a
r
c
h
 
C
V
 
s
c
o
r
e
 
o
n
 
t
e
s
t
 
s
e
t
:
 
{
0
:
0
.
4
f
}
'
.
f
o
r
m
a
t
(
g
r
i
d
_
s
e
a
r
c
h
.
s
c
o
r
e
(
X
_
t
e
s
t
,
 
y
_
t
e
s
t
)
)
)
```

#
#
#
 
C
o
m
m
e
n
t
s






-
 
O
u
r
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
 
t
e
s
t
 
a
c
c
u
r
a
c
y
 
i
s
 
0
.
8
5
0
1
 
w
h
i
l
e
 
G
r
i
d
S
e
a
r
c
h
 
C
V
 
a
c
c
u
r
a
c
y
 
i
s
 
0
.
8
5
0
7
.






-
 
W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
G
r
i
d
S
e
a
r
c
h
 
C
V
 
i
m
p
r
o
v
e
 
t
h
e
 
p
e
r
f
o
r
m
a
n
c
e
 
f
o
r
 
t
h
i
s
 
p
a
r
t
i
c
u
l
a
r
 
m
o
d
e
l
.


#
 
*
*
2
1
.
 
R
e
s
u
l
t
s
 
a
n
d
 
c
o
n
c
l
u
s
i
o
n
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
2
1
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)


1
.
	
T
h
e
 
l
o
g
i
s
t
i
c
 
r
e
g
r
e
s
s
i
o
n
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
i
s
 
0
.
8
5
0
1
.
 
S
o
,
 
t
h
e
 
m
o
d
e
l
 
d
o
e
s
 
a
 
v
e
r
y
 
g
o
o
d
 
j
o
b
 
i
n
 
p
r
e
d
i
c
t
i
n
g
 
w
h
e
t
h
e
r
 
o
r
 
n
o
t
 
i
t
 
w
i
l
l
 
r
a
i
n
 
t
o
m
o
r
r
o
w
 
i
n
 
A
u
s
t
r
a
l
i
a
.




2
.
	
S
m
a
l
l
 
n
u
m
b
e
r
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
 
p
r
e
d
i
c
t
 
t
h
a
t
 
t
h
e
r
e
 
w
i
l
l
 
b
e
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.
 
M
a
j
o
r
i
t
y
 
o
f
 
o
b
s
e
r
v
a
t
i
o
n
s
 
p
r
e
d
i
c
t
 
t
h
a
t
 
t
h
e
r
e
 
w
i
l
l
 
b
e
 
n
o
 
r
a
i
n
 
t
o
m
o
r
r
o
w
.




3
.
	
T
h
e
 
m
o
d
e
l
 
s
h
o
w
s
 
n
o
 
s
i
g
n
s
 
o
f
 
o
v
e
r
f
i
t
t
i
n
g
.




4
.
	
I
n
c
r
e
a
s
i
n
g
 
t
h
e
 
v
a
l
u
e
 
o
f
 
C
 
r
e
s
u
l
t
s
 
i
n
 
h
i
g
h
e
r
 
t
e
s
t
 
s
e
t
 
a
c
c
u
r
a
c
y
 
a
n
d
 
a
l
s
o
 
a
 
s
l
i
g
h
t
l
y
 
i
n
c
r
e
a
s
e
d
 
t
r
a
i
n
i
n
g
 
s
e
t
 
a
c
c
u
r
a
c
y
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
a
 
m
o
r
e
 
c
o
m
p
l
e
x
 
m
o
d
e
l
 
s
h
o
u
l
d
 
p
e
r
f
o
r
m
 
b
e
t
t
e
r
.




5
.
	
I
n
c
r
e
a
s
i
n
g
 
t
h
e
 
t
h
r
e
s
h
o
l
d
 
l
e
v
e
l
 
r
e
s
u
l
t
s
 
i
n
 
i
n
c
r
e
a
s
e
d
 
a
c
c
u
r
a
c
y
.




6
.
	
R
O
C
 
A
U
C
 
o
f
 
o
u
r
 
m
o
d
e
l
 
a
p
p
r
o
a
c
h
e
s
 
t
o
w
a
r
d
s
 
1
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
o
u
r
 
c
l
a
s
s
i
f
i
e
r
 
d
o
e
s
 
a
 
g
o
o
d
 
j
o
b
 
i
n
 
p
r
e
d
i
c
t
i
n
g
 
w
h
e
t
h
e
r
 
i
t
 
w
i
l
l
 
r
a
i
n
 
t
o
m
o
r
r
o
w
 
o
r
 
n
o
t
.




7
.
	
O
u
r
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
i
s
 
0
.
8
5
0
1
 
w
h
e
r
e
a
s
 
a
c
c
u
r
a
c
y
 
s
c
o
r
e
 
a
f
t
e
r
 
R
F
E
C
V
 
i
s
 
0
.
8
5
0
0
.
 
S
o
,
 
w
e
 
c
a
n
 
o
b
t
a
i
n
 
a
p
p
r
o
x
i
m
a
t
e
l
y
 
s
i
m
i
l
a
r
 
a
c
c
u
r
a
c
y
 
b
u
t
 
w
i
t
h
 
r
e
d
u
c
e
d
 
s
e
t
 
o
f
 
f
e
a
t
u
r
e
s
.




8
.
	
I
n
 
t
h
e
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
,
 
w
e
 
h
a
v
e
 
F
P
 
=
 
1
1
7
5
 
w
h
e
r
e
a
s
 
F
P
1
 
=
 
1
1
7
4
.
 
S
o
,
 
w
e
 
g
e
t
 
a
p
p
r
o
x
i
m
a
t
e
l
y
 
s
a
m
e
 
n
u
m
b
e
r
 
o
f
 
f
a
l
s
e
 
p
o
s
i
t
i
v
e
s
.
 
A
l
s
o
,
 
F
N
 
=
 
3
0
8
7
 
w
h
e
r
e
a
s
 
F
N
1
 
=
 
3
0
9
1
.
 
S
o
,
 
w
e
 
g
e
t
 
s
l
i
g
h
l
y
 
h
i
g
h
e
r
 
f
a
l
s
e
 
n
e
g
a
t
i
v
e
s
.




9
.
	
O
u
r
,
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
 
s
c
o
r
e
 
i
s
 
f
o
u
n
d
 
t
o
 
b
e
 
0
.
8
4
7
6
.
 
T
h
e
 
a
v
e
r
a
g
e
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
s
c
o
r
e
 
i
s
 
0
.
8
4
7
4
.
 
S
o
,
 
w
e
 
c
a
n
 
c
o
n
c
l
u
d
e
 
t
h
a
t
 
c
r
o
s
s
-
v
a
l
i
d
a
t
i
o
n
 
d
o
e
s
 
n
o
t
 
r
e
s
u
l
t
 
i
n
 
p
e
r
f
o
r
m
a
n
c
e
 
i
m
p
r
o
v
e
m
e
n
t
.




1
0
.
	
O
u
r
 
o
r
i
g
i
n
a
l
 
m
o
d
e
l
 
t
e
s
t
 
a
c
c
u
r
a
c
y
 
i
s
 
0
.
8
5
0
1
 
w
h
i
l
e
 
G
r
i
d
S
e
a
r
c
h
 
C
V
 
a
c
c
u
r
a
c
y
 
i
s
 
0
.
8
5
0
7
.
 
W
e
 
c
a
n
 
s
e
e
 
t
h
a
t
 
G
r
i
d
S
e
a
r
c
h
 
C
V
 
i
m
p
r
o
v
e
 
t
h
e
 
p
e
r
f
o
r
m
a
n
c
e
 
f
o
r
 
t
h
i
s
 
p
a
r
t
i
c
u
l
a
r
 
m
o
d
e
l
.




#
 
*
*
2
2
.
 
R
e
f
e
r
e
n
c
e
s
*
*
 
<
a
 
c
l
a
s
s
=
"
a
n
c
h
o
r
"
 
i
d
=
"
2
2
"
>
<
/
a
>






[
T
a
b
l
e
 
o
f
 
C
o
n
t
e
n
t
s
]
(
#
0
.
1
)








T
h
e
 
w
o
r
k
 
d
o
n
e
 
i
n
 
t
h
i
s
 
p
r
o
j
e
c
t
 
i
s
 
i
n
s
p
i
r
e
d
 
f
r
o
m
 
f
o
l
l
o
w
i
n
g
 
b
o
o
k
s
 
a
n
d
 
w
e
b
s
i
t
e
s
:
-






1
.
 
H
a
n
d
s
 
o
n
 
M
a
c
h
i
n
e
 
L
e
a
r
n
i
n
g
 
w
i
t
h
 
S
c
i
k
i
t
-
L
e
a
r
n
 
a
n
d
 
T
e
n
s
o
r
f
l
o
w
 
b
y
 
A
u
r
e
́
l
i
e
́
n
 
G
e
́
r
o
n




2
.
 
I
n
t
r
o
d
u
c
t
i
o
n
 
t
o
 
M
a
c
h
i
n
e
 
L
e
a
r
n
i
n
g
 
w
i
t
h
 
P
y
t
h
o
n
 
b
y
 
A
n
d
r
e
a
s
 
C
.
 
M
u
̈
l
l
e
r
 
a
n
d
 
S
a
r
a
h
 
G
u
i
d
o




3
.
 
U
d
e
m
y
 
c
o
u
r
s
e
 
–
 
M
a
c
h
i
n
e
 
L
e
a
r
n
i
n
g
 
–
 
A
 
Z
 
b
y
 
K
i
r
i
l
l
 
E
r
e
m
e
n
k
o
 
a
n
d
 
H
a
d
e
l
i
n
 
d
e
 
P
o
n
t
e
v
e
s




4
.
 
U
d
e
m
y
 
c
o
u
r
s
e
 
–
 
F
e
a
t
u
r
e
 
E
n
g
i
n
e
e
r
i
n
g
 
f
o
r
 
M
a
c
h
i
n
e
 
L
e
a
r
n
i
n
g
 
b
y
 
S
o
l
e
d
a
d
 
G
a
l
l
i




5
.
 
U
d
e
m
y
 
c
o
u
r
s
e
 
–
 
F
e
a
t
u
r
e
 
S
e
l
e
c
t
i
o
n
 
f
o
r
 
M
a
c
h
i
n
e
 
L
e
a
r
n
i
n
g
 
b
y
 
S
o
l
e
d
a
d
 
G
a
l
l
i




6
.
 
h
t
t
p
s
:
/
/
e
n
.
w
i
k
i
p
e
d
i
a
.
o
r
g
/
w
i
k
i
/
L
o
g
i
s
t
i
c
_
r
e
g
r
e
s
s
i
o
n




7
.
 
h
t
t
p
s
:
/
/
m
l
-
c
h
e
a
t
s
h
e
e
t
.
r
e
a
d
t
h
e
d
o
c
s
.
i
o
/
e
n
/
l
a
t
e
s
t
/
l
o
g
i
s
t
i
c
_
r
e
g
r
e
s
s
i
o
n
.
h
t
m
l




8
.
 
h
t
t
p
s
:
/
/
e
n
.
w
i
k
i
p
e
d
i
a
.
o
r
g
/
w
i
k
i
/
S
i
g
m
o
i
d
_
f
u
n
c
t
i
o
n




9
.
 
h
t
t
p
s
:
/
/
w
w
w
.
s
t
a
t
i
s
t
i
c
s
s
o
l
u
t
i
o
n
s
.
c
o
m
/
a
s
s
u
m
p
t
i
o
n
s
-
o
f
-
l
o
g
i
s
t
i
c
-
r
e
g
r
e
s
s
i
o
n
/




1
0
.
 
h
t
t
p
s
:
/
/
w
w
w
.
k
a
g
g
l
e
.
c
o
m
/
m
n
a
s
s
r
i
b
/
t
i
t
a
n
i
c
-
l
o
g
i
s
t
i
c
-
r
e
g
r
e
s
s
i
o
n
-
w
i
t
h
-
p
y
t
h
o
n




1
1
.
 
h
t
t
p
s
:
/
/
w
w
w
.
k
a
g
g
l
e
.
c
o
m
/
n
e
i
s
h
a
/
h
e
a
r
t
-
d
i
s
e
a
s
e
-
p
r
e
d
i
c
t
i
o
n
-
u
s
i
n
g
-
l
o
g
i
s
t
i
c
-
r
e
g
r
e
s
s
i
o
n




1
2
.
 
h
t
t
p
s
:
/
/
w
w
w
.
r
i
t
c
h
i
e
n
g
.
c
o
m
/
m
a
c
h
i
n
e
-
l
e
a
r
n
i
n
g
-
e
v
a
l
u
a
t
e
-
c
l
a
s
s
i
f
i
c
a
t
i
o
n
-
m
o
d
e
l
/




S
o
,
 
n
o
w
 
w
e
 
w
i
l
l
 
c
o
m
e
 
t
o
 
t
h
e
 
e
n
d
 
o
f
 
t
h
i
s
 
k
e
r
n
e
l
.




I
 
h
o
p
e
 
y
o
u
 
f
i
n
d
 
t
h
i
s
 
k
e
r
n
e
l
 
u
s
e
f
u
l
 
a
n
d
 
e
n
j
o
y
a
b
l
e
.




Y
o
u
r
 
c
o
m
m
e
n
t
s
 
a
n
d
 
f
e
e
d
b
a
c
k
 
a
r
e
 
m
o
s
t
 
w
e
l
c
o
m
e
.




T
h
a
n
k
 
y
o
u




[
G
o
 
t
o
 
T
o
p
]
(
#
0
)

