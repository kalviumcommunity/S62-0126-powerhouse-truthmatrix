# Milestone 4.16 – Conditional Statements Demo

# 1. Basic if statement
age = 20

print("Basic if example:")
if age >= 18:
    print("You are eligible to vote")
print()


# 2. if–else example
marks = 45

print("if-else example:")
if marks >= 50:
    print("You passed")
else:
    print("You failed")
print()


# 3. if–elif–else example
score = 82

print("if-elif-else example:")
if score >= 90:
    print("Grade: A")
elif score >= 75:
    print("Grade: B")
elif score >= 50:
    print("Grade: C")
else:
    print("Grade: Fail")
print()


# 4. Logical operators example
temperature = 30
is_raining = False

print("Logical operators example:")
if temperature > 25 and not is_raining:
    print("Good weather to go outside")
elif temperature > 25 and is_raining:
    print("It's hot but raining")
else:
    print("Weather is not suitable")
print()


# 5. OR operator example
day = "Saturday"

print("OR operator example:")
if day == "Saturday" or day == "Sunday":
    print("It's the weekend!")
else:
    print("It's a weekday")