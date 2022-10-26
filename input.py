lps/kmp, knapsack, coin change
dp part 1 revision

correct all algorithms

.copy() to copy a structure

l=list(map(int, input().strip().split()))
m=input()
map(lambda a: a**2, nums)
nums+nums
l.insert(i, x)
for a, b in zip(nums, index):
ord()
chr()
[chr(x) for x in range(97, 123)]

l.remove(4)
l.index(2)

del l[4]
del l[4:]

# sort a list
l.sort()
sorted(a)
# reverse a list
l.reverse()
reversed(l)
s[::-1]
# iterate over key value
for x,y in d.items()
# get from dict default
d.get(key, default_return)

import bisect
bisect.bisect(l, n, begin_index, end_index)

##################################### heap, priority queue #####################################
# for priority queue use tuples in heapq
import heapq
li = [5, 7, 9, 1, 3]
heapq.heapify(li)
print (list(li))
heapq.heappush(li,4)
x=heapq.heappop(li)
heapq.nlargest(n, li)
heapq.nsmallest(n, li)
c=heapq.merge([1,2,3],[4,5,6])

  
##################################### stack #####################################
from queue import LifoQueue
stack = LifoQueue(maxsize=3)
stack.qsize()
stack.put('a')
stack.full()
stack.get()
stack.empty()

# using list
stack=[]
len(stack)
stack.append('a')
stack[-1]
stack.pop()
len(stack)==0

##################################### queue ####################################
from queue import Queue
q = Queue(maxsize = 3)
q.qsize()
q.put('a')
q.full()
q.get()
q.empty()

##################################### sets #####################################
a = set(["a", "b", "c"])
b = {"Jay", "Idrish", "Archil"}
a.add("d")
a.remove("d")
a.pop()
u = a.union(b)
i = a.intersection(b)
d = a - b

key in s	    containment check
key not in s    on-containment check
s1 == s2	    s1 is equivalent to s2
s1 != s2	    s1 is not equivalent to s2
s1 <= s2	    s1 is subset of s2
s1 < s2	        s1 is proper subset of s2
s1 >= s2	    s1 is superset of s2
s1 > s2	        s1 is proper superset of s2
s1 | s2	        the union of s1 and s2
s1 & s2	        the intersection of s1 and s2
s1 – s2	        the set of elements in s1 but not s2
s1 ˆ s2	        the set of elements in precisely one of s1 or s2


############################################################################################
################################# Algorithms ###############################################
############################################################################################

##################################### binary search #####################################
s=0
e=n-1
while s<=e:
    m=int((s+e)/2)
    if nums[m]==target:
        return m
    elif nums[m]>target:
        e=m-1
    else:
        s=m+1

##################################### binary search rotated array #####################################
n=len(nums)
if n==0:
    return -1
s=0
e=n-1
while s<=e:
    m=int((s+e)/2)
    if nums[m]==target:
        return m
    elif nums[s]<=nums[m]:
        if nums[m]<target or target<nums[s]:
            s=m+1
        else:
            e=m-1
    else:
        if nums[s]<=target or target<nums[m]:
            e=m-1
        else:
            s=m+1
if s>n-1 or nums[s]!=target:
    return -1
else:
    return s

##################################### Disjoint Set #####################################
def findParent(x):
    i=x
    while P[i] != i:
        i=P[i]
    j=x
    while P[j] != j:
        t=j
        j=P[j]
        P[t]=i
    return i

def union(x, y):
    x=findParent(u)
    y=findParent(v)
    if x==y:
        return
    if R[x]<R[y]:
        P[x]=y
    elif R[x]>R[y]:
        P[y] = x
    else:
        P[y] = x
        R[x] += 1

R=[0]*n
P=list(range(n))

for u, v in E:
    union(u, v)
C=set()
for p in P:
    C.add(p)
return len(C)

##################################### Unbounded Knapsack Problem #####################################
P=[1,2,5,6]
W=[2,3,4,5]
T=8
n=len(P)

dp=[0]*(T+1)
pick=[0]*(T+1)
for i in range(n):
    for w in range(W[i], T+1):
        if dp[w]<dp[w-W[i]]+P[i]:
            dp[w]=dp[w-W[i]]+P[i]
            pick[w]=i
B=[0]*n
i=T
while pick[i]!=0:
    B[pick[i]]+=1
    i-=W[pick[i]]
print(B)
print(dp[T])

##################################### Fractional Knapsack Problem #####################################
from queue import PriorityQueue

P=[10,5,15,7,6,18,3]
W=[2,3,5,7,1,4,1]
T=15
n=len(P)

F=[0]*n
K=0
Q=PriorityQueue()

_=[Q.put((-p/w, i)) for i, (p, w) in enumerate(zip(P,W))]

while not Q.empty():
    _, i=Q.get()
    if T>W[i]:
        K+=P[i]
        F[i]=1
        T-=W[i]
    elif T>0:
        K+=P[i]*T/W[i]
        F[i]=T/W[i]
        T=0
        break

print(F)
print(K)

##################################### Longest Common Subsequence #####################################
# shortest common supersequence
# longest common substring
# longest palindromic subsequence

s='aaaataaaaa'
t='aaaasaaaaa'
m=len(s)
n=len(t)

dp=[0]*(m+1)

for c in t:
    temp=dp
    for i in range(m, 0, -1):
        if c==s[i-1]:
            dp[i]=temp[i-1]+1
        else:
            dp[i]=max(dp[i-1], temp[i])
print(dp[m])
lcs=''
for i in range(m):
    if dp[i+1]-dp[i]==1:
        lcs+=s[i]
print(lcs)

##################################### Longest Increasing Subsequence #####################################
# maximum height in boxes-produce all rotations, sort in asc, find LIS
l=[5,8,7,1,9]
n=len(l)

dp=[1]*n

for i in range(n):
    for j in range(i):
        if l[j]<l[i]:
            dp[i]=max(dp[i], dp[j]+1)
m=dp[-1]
lis=[]

for i in range(n-1,-1,-1):
    if m==0:
        break
    if dp[i]==m:
        lis.append(l[i])
        m-=1
lis=list(reversed(lis))
print(lis)

##################################### Largest Sum Contiguous Subarray(Kandane's Algorithm) #####################################
import math
l=[-2, -3, 4, -1, -2, 1, 5, -3]
n=len(l)

msf=-math.inf
meh=0
s=0
for i in range(n):
    meh=meh+l[i]
    if msf<meh:
        msf=meh
        start=s
        end=i
    if meh<0:
        meh=0
        s=i+1
print(msf)
print(start)
print(end)

##################################### Largest Product Contiguous Subarray #####################################


##################################### Quick Select #####################################
l = [1, 3, 5, 2, 8, 4]
n = len(l)

start = 0
end = n-1
k = 3
while True:
    s = start
    e = end
    p = l[end]

    pivot = s
    pivot_value = l[e]
    for i in range(s, e):
      if l[i] <= pivot_value:
        l[i], l[pivot] = l[pivot], l[i]
        pivot += 1

    l[e], l[pivot] = l[pivot], l[e]

    if pivot+1 < k:
        start = pivot+1
    elif pivot+1 > k:
        end = pivot-1
    else:
        print(l[pivot])
        break

print(l)


##################################### Longest Palindrome Substring #####################################
longest=""
m=1
n=len(s)
start=0
for i in range(1,n-1):
    low=high=i
    while low>=0 and high<n and s[low]==s[high]:
        if high-low+1>m:
            start=low
            m=high-low+1
        low-=1
        high+=1
for i in range(0,n-1):
    low=i
    high=i+1
    while low>=0 and high<n and s[low]==s[high]:
        if high-low+1>m:
            start=low
            m=high-low+1
        low-=1
        high+=1
return s[start:start+m]

##################################### LPS/KMP #####################################
def getlps(p):
    n=len(p)
    lps=[0]*n

    lps[0]=0
    l=0
    i=1
    while i<n:
        if p[i]==p[l]:
            l+=1
            lps[i]=l
            i+=1
        else:
            if l==0:
                lps[i]=0
                i+=1
            else:
                l=lps[l-1]
    return lps

s='AABAACAADAABAABA'
p='AABA'
lps=getlps(p)

i=0
j=0
n=len(s)
m=len(p)
ans=[]

while i<n:
    if s[i]==p[j]:
        i+=1
        j+=1
    else:
        if j==0:
            i+=1
        else:
            j=lps[j-1]
    if j==m:
        ans.append(i-j)
        j=lps[j-1]

print(ans)


