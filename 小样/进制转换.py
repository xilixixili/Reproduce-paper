def data1(m,x):
    lenth=len(x)
    res=0
    for i in range(lenth):
        res+=int(list1.index(x[i]))*m**(lenth-1-i)
    return res
list1=['0','1','2','3','4','5','6','7','8','9',
      'A','B','C','D','E','F',
      'G','H','I','J','K','L',
      'M','N','O','P','Q','R',
      'S','T','U','V','W','X',
      'Y','Z']


def root(x , y , k) :
    temp = 1
    while y :
        # 相当于y%2
        if y & 1 == 1 :  # 当是奇数时
            temp = (temp * x) % k
        x = (x * x) % k
        y = y >> 1
        temp = temp if temp else k
    return temp


if __name__ == '__main__' :
    while True :
        try :
            x , y , k = map(int , input().strip().split())
            print(root(x , y , k - 1))

        except :
            break