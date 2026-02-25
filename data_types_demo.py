# Numeric Data Types
a = 10        # integer
b = 3.5       # float

print("Integer value:", a)
print("Float value:", b)
print("Addition:", a + b)
print("Division:", a / 2)

# String Data Types
name = "Karan"
course = "Python Fundamentals"

print("Name:", name)
print("Course:", course)

# String Concatenation
message = name + " is learning " + course
print(message)

# Mixing Types (will cause error if not converted)
age = 20

# Correct way
print("Age:", str(age))

# Convert string to number
num_str = "25"
num = int(num_str)
print("Converted number + 5 =", num + 5)

# Checking types
print(type(a))
print(type(b))
print(type(name))
print(type(num))