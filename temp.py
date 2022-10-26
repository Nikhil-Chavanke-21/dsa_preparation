# from queue import PriorityQueue

# class node:
# 	def __init__(self, freq, symbol, left=None, right=None):
# 		self.symbol = symbol
# 		self.left = left
# 		self.right = right

# C=['a', 'b', 'c', 'd', 'e', 'f']
# F=[ 5, 9, 12, 13, 16, 45]
# nodes=[node(f, c) for f,c in zip(F,C)]

# q=PriorityQueue()
# for f, c in zip(F, C):
#     q.put(f, node(f, c))

# while q.qsize()>1:
# 	fl, l=q.get()
# 	fr, r=q.get()

# 	new=node(fl+fr, left.symbol+right.symbol, left, right)

# 	# remove the 2 nodes and add their
# 	# parent as new node among others
# 	nodes.remove(left)
# 	nodes.remove(right)
# 	nodes.append(newNode)

# printNodes(nodes[0])
