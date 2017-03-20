class A:
    def __init__(self,x):
        self.X=x
        print(self.X)

a=A(1)
print(a.X)
a.__init__(2)
print(a.X)