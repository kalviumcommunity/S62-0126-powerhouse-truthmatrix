# Milestone 4.15 - Lists, Tuples, Dictionaries

print("---- LIST EXAMPLE ----")
# List (mutable)
fruits = ["apple", "banana", "mango"]
print("Original list:", fruits)

# Access
print("First fruit:", fruits[0])

# Modify
fruits[1] = "orange"
print("After modification:", fruits)

# Add
fruits.append("grapes")
print("After adding:", fruits)

# Remove
fruits.remove("apple")
print("After removing:", fruits)


print("\n---- TUPLE EXAMPLE ----")
# Tuple (immutable)
numbers = (10, 20, 30)
print("Tuple:", numbers)

# Access
print("First number:", numbers[0])

# Immutability demonstration
try:
    numbers[1] = 50
except TypeError as e:
    print("Error (Tuples are immutable):", e)


print("\n---- DICTIONARY EXAMPLE ----")
# Dictionary (key-value pairs)
student = {
    "name": "Karan",
    "age": 20,
    "course": "Engineering"
}

print("Student dictionary:", student)

# Access value
print("Name:", student["name"])

# Modify value
student["age"] = 21

# Add new key
student["city"] = "Bangalore"

print("Updated dictionary:", student)