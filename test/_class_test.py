class Animal(object):  
   def __init__(self, name, age):
       self.name = name
       self.age = age
       self.pp = 5
       
   def call(self):
       print(self.name, '会叫')

class Cat(Animal):
   def __init__(self, name, age, sex):
       super(Cat, self).__init__(name, age)
       self.sex = sex

    #def call(self):
    #    print(self.name, '会“喵喵”叫')

class Dog(Animal):
   def __init__(self, name, age, sex):
       super(Dog, self).__init__(name, age)
       self.sex = sex
   def call(self):
       print(self.name, '会“汪汪”叫')
       



def do(all):
   all.call()

A = Animal('小黑',4)
C = Cat('喵喵', 2, '男')
D = Dog('旺财', 5, '女')

for x in (A,C,D):
   do(x)
   

print(A.pp)
print(C.pp)
   
