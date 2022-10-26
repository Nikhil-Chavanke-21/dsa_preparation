def max_profit(a):
  max_sum = 0
  max_end_here = 0
  if len(a)<=1:
    return 0
  for i in range(len(a)-1):
    delta=a[i+1]-a[i]
    max_end_here+=delta
    max_end_here=max(0, max_end_here)
    max_sum = max(max_end_here, max_sum)
  return max_sum

def test(array, exp):
  if max_profit(array)==exp:
    print('Passed!!')
  else:
    print('Failed')


test([4, 11, 3, 5, 2, 8, 7], 7)
test([4, 9, 3, 5, 2, 8, 7], 6)

