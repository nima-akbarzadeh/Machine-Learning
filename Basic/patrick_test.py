import json
import random
import sys

import numpy as np
from collections import Counter, deque
from itertools import product, permutations, combinations, combinations_with_replacement, groupby


if __name__ == '__main__':

    print('====================================================================================')
    print('LISTS ==============================================================================')
    my_list = list(np.arange(10))
    print(f"my_list = {my_list}")
    print('------------------------------------------------------------------------------------')

    # Select a subset of list
    ini_index = 1
    step_size = 2
    end_index = 9
    my_list_1 = my_list[ini_index:end_index:step_size]
    print(f"Subset list is {my_list_1}")
    print('------------------------------------------------------------------------------------')

    # Add a list to another list
    list_1 = [1, 2, 3]
    list_2 = [4, 5, 6]
    my_list = list_1 + list_2
    print(f"New combined list is {my_list}")
    print('------------------------------------------------------------------------------------')

    # Count the number of elements in a list
    my_list = list(np.arange(10)) + list(np.arange(5)) + list(np.arange(2))
    for n in range(10):
        print(f"Number of elements with value {n} is {my_list.count(n)}")
    print('------------------------------------------------------------------------------------')

    print('====================================================================================')
    print('TUPLE ==============================================================================')
    my_tuple = tuple(np.arange(10))
    print(f"my_tuple = {my_tuple}")

    # Select a subset of tuple
    ini_index = 1
    step_size = 2
    end_index = 9
    my_tuple_1 = my_tuple[ini_index:end_index:step_size]
    print(f"Subset tuple is {my_tuple_1}")
    print('------------------------------------------------------------------------------------')

    # Count the number of elements in a list
    my_tuple = tuple(list(np.arange(10)) + list(np.arange(5)) + list(np.arange(2)))
    for n in range(10):
        print(f"Number of elements with value {n} is {my_tuple.count(n)}")
    print('------------------------------------------------------------------------------------')

    print('====================================================================================')
    print('DICTIONARY =========================================================================')

    my_dict = {'a': 1, 'b': 2, 'c': 3}
    print(my_dict)

    # Delete elements 'b' & 'c'
    my_dict.pop('b')
    print(my_dict)
    del my_dict['c']
    print(my_dict)

    print(my_dict['a'])
    print(my_dict.get('a'))
    print('looks the same ...')
    print('What if you had KeyError!?')
    print(my_dict.get('b'))  # --> Returns None
    # print(my_dict['b']) --> Throws KeyError

    # Add new dictionaries to the old one (update, and **kwargs)

    original_dict = {'a': 1, 'b': 2}
    dict1 = {'c': 3}
    dict2 = {'d': 4, 'e': 5}

    print(original_dict.update(dict1))
    print(original_dict.update(dict2))

    original_dict = {'a': 1, 'b': 2}
    dict1 = {'c': 3}
    dict2 = {'d': 4, 'e': 5}

    new_dict = {**original_dict, **dict1, **dict2}

    print(new_dict)

    print('====================================================================================')
    print('SET ================================================================================')

    # Unique values in a set
    my_set = {10, 20, 30, 10, 20}
    print(my_set)

    # Remove an element
    element_to_remove = 10
    my_set.remove(element_to_remove)
    print(my_set)
    my_set.add(element_to_remove)
    my_set.discard(element_to_remove)
    print(my_set)

    # Create a set and augment
    my_set = set()
    print(my_set)
    my_set.add(10)
    my_set.add(20)
    my_set.add(30)
    print(my_set)

    # Set operations
    my_set = {10, 20, 30}
    your_set = {20, 30, 40}
    their_set = {30, 40, 50}
    my_intersect_yours = my_set.intersection(your_set)
    print(f'intersection of mine and yours is {my_intersect_yours}')
    my_diff_yours = my_set.difference(your_set)
    print(f'difference of mine from yours is {my_diff_yours}')
    my_union_theirs = my_set.union(their_set)
    print(f'union of mine and them is {my_union_theirs}')
    print(f'remaining elements of mine and yours if common elements are removed is {my_set.symmetric_difference(your_set)}')
    my_union_yours = my_set.union(your_set)
    my_common_yours = my_intersect_yours
    print(f'remaining elements of mine and yours if common elements are removed is {my_union_yours.difference(my_common_yours)}')

    # Check if a set is subset, superset or disjoint from another one
    print(f'check subset {my_set.issubset(your_set)}')
    print(f'check superset {my_set.issuperset(your_set)}')
    print(f'check disjoint {my_set.isdisjoint(your_set)}')

    print(f"unmutable frozen set is {frozenset([1, 2, 2, 3])}")

    print('====================================================================================')
    print('STRINGS ============================================================================')

    print('STRINGS are immutable')

    my_text = '   hello world   '
    print(f"removing space before and after doesn't work like this: {my_text.strip()}")
    new_text = my_text.strip()
    print(f'removing space before and after: {new_text}')

    starting = 'hello'
    print(f"Does it start with {starting}: {new_text.startswith(starting)}")
    ending = 'world'
    print(f"Does it end with {ending}: {new_text.endswith(ending)}")

    print(f"count l: {new_text.count('l')}")
    print(f"count lo: {new_text.count('lo')}")
    print(f"count pp: {new_text.count('pp')}")
    print(f"find l: {new_text.find('l')}")
    print(f"find lo: {new_text.find('lo')}")
    print(f"find pp: {new_text.find('pp')}")

    new_text = new_text.replace('world', 'universe')
    print(f"new text: {new_text}")

    delimiter = ' '
    text_list = new_text.split(delimiter)
    print(f"list of words: {text_list}")
    orig_text = delimiter.join(text_list)
    print(f"orig new text: {orig_text}")

    print('====================================================================================')
    print('COUNTER ============================================================================')

    # Count the number of elements in any iterable, get the most counted one
    any_iterable = 'aaabbbbcccc'

    count_dict = Counter(any_iterable)
    print(count_dict)
    print(count_dict.most_common())
    print(count_dict.most_common(1))
    print(count_dict.most_common(1)[0][0])
    print(count_dict.most_common(1)[0][1])

    print('====================================================================================')
    print('DEQUE ==============================================================================')

    # Count the number of elements in any iterable, get the most counted one
    any_iterable = 'a a a b b b b c c c c'.split(' ')
    print(any_iterable)
    any_iterable = deque(any_iterable)
    print(any_iterable)

    any_iterable.append('q')
    print(any_iterable)
    any_iterable.appendleft('q')
    print(any_iterable)
    any_iterable.pop()
    print(any_iterable)
    any_iterable.popleft()
    print(any_iterable)

    print('====================================================================================')
    print('PRODUCT ============================================================================')

    list1 = [1, 2]
    list2 = ['a', 'b']

    # Compute the Cartesian product
    result = product(list1, list2)
    print('Product of 2 sets: ')
    for item in result:
        print(item)

    list3 = ['x', 'y']
    result = product(list1, list2, list3)

    print('Product of 3 sets: ')
    for item in result:
        print(item)

    print('Repeat 3 times: ')
    result = product(list1, repeat=3)
    for item in result:
        print(item)

    print('====================================================================================')
    print('PERMUTATIONS =======================================================================')

    list1 = [1, 2, 3, 4]

    # Compute all permutations
    result = permutations(list1)
    print(f'Permutations of the set {list1}: ')
    for item in result:
        print(item)

    # Compute all permutations by a subset of two
    result = permutations(list1, 2)
    print(f'Permutations of the set {list1} with 2 elements: ')
    for item in result:
        print(item)

    print('====================================================================================')
    print('COMBINATIONS =======================================================================')

    list1 = [1, 2, 3, 4]

    # Compute all permutations by a subset of two
    result = combinations(list1, 2)
    print(f'Combinations of the set {list1} with 2 elements: ')
    for item in result:
        print(item)

    # Compute all combinations
    result = combinations_with_replacement(list1, 2)
    print(f'Combinations with replacement of the set {list1}: ')
    for item in result:
        print(item)

    print('====================================================================================')
    print('GROUPBY =======================================================================')

    my_list = list(np.arange(10))
    print(my_list)

    print('You have a list and apply a function on it and group by the result of the function:')
    def my_func(x):
        return x < 5

    result = groupby(my_list, key=my_func)
    for key, values in result:
        print(f"{key}: {list(values)}")

    print('You have a list and group them by similar keys:')
    my_persons = [
        {'name': 'n1', 'love': 'm1', 'city': 'montreal'},
        {'name': 'n2', 'love': 'm2', 'city': 'montreal'},
        {'name': 'n3', 'love': 'm3', 'city': 'shiraz'},
        {'name': 'n4', 'love': 'm4', 'city': 'tehran'},
    ]
    result = groupby(my_persons, key=lambda x: x['city'])
    for key, values in result:
        print(f"{key}: {list(values)}")

    print('====================================================================================')
    print('LAMBDA =============================================================================')

    my_list = list(np.arange(10))
    print(my_list)
    print('You have a list and apply a function on it and group by the result of the function using LAMBDA:')
    result = groupby(my_list, key=lambda x: x > 5)
    for key, values in result:
        print(f"{key}: {list(values)}")

    # Sort this list based on the second element
    my_list = [(1, -2), (-2, 3), (3, -4), (-4, 5)]
    print(my_list)
    print(sorted(my_list))
    print(sorted(my_list, key=lambda x: x[1]))

    # Keep elements where the second value is negative using filter and lambda
    new_list = list(filter(lambda x: x[1] < 0, my_list))
    print(new_list)

    # Keep elements where the second value is negative using list iterations
    new_list = [x for x in my_list if x[1] < 0]
    print(new_list)

    print('====================================================================================')
    print('EXCEPTIONS =========================================================================')

    print("Common types:")
    print("SyntaxError: This error occurs when the code is not written in proper Python syntax.")
    try:
        exec("if True print('This will cause a SyntaxError')")
    except SyntaxError as e:
        print(f"Caught a SyntaxError: {e}")
    print("NameError: This error happens when the code references a variable or function name that hasn't been defined.")
    try:
        print(unknown_variable)
    except NameError as e:
        print(f"Caught a NameError: {e}")
    print("TypeError: This error occurs when an operation is applied to an object of inappropriate type.")
    try:
        result = "The number is: " + 5
    except TypeError as e:
        print(f"Caught a TypeError: {e}")
    print("IndexError: This error happens when trying to access an index that is out of range for a sequence (e.g., list, string, or tuple).")
    try:
        numbers = [1, 2, 3]
        print(numbers[5])
    except IndexError as e:
        print(f"Caught an IndexError: {e}")
    print("KeyError: This error occurs when trying to access a dictionary with a key that doesn't exist.")
    try:
        my_dict = {"name": "Alice", "age": 30}
        print(my_dict["address"])
    except KeyError as e:
        print(f"Caught a KeyError: {e}")
    print("AttributeError: This error happens when trying to access an attribute or method that doesn't exist for an object.")
    try:
        number = 10
        number.append(5)
    except AttributeError as e:
        print(f"Caught an AttributeError: {e}")
    print("ValueError: This error occurs when a function receives an argument of the correct type but an inappropriate value.")
    try:
        int_value = int("not_a_number")
    except ValueError as e:
        print(f"Caught a ValueError: {e}")
    print("ImportError: This error happens when an import statement fails to find the module or a name within the module.")
    try:
        import non_existent_module
    except ImportError as e:
        print(f"Caught an ImportError: {e}")
    print("ZeroDivisionError: This error occurs when a division or modulo operation is attempted with a denominator of zero.")
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        print(f"Caught a ZeroDivisionError: {e}")
    print("FileNotFoundError: This error happens when trying to open a file that does not exist.")
    try:
        with open("non_existent_file.txt", "r") as file:
            content = file.read()
    except FileNotFoundError as e:
        print(f"Caught a FileNotFoundError: {e}")
    print("StopIteration for Generators")

    print('====================================================================================')
    print('JSON ===============================================================================')

    nima_dict = {
        'last_name': 'Akbarzadeh',
        'married': True,
        'love': 'Mina',
        'city': 'Shiraz',
        'age': 32,
    }
    print(nima_dict)
    nima_json = json.dumps(nima_dict, indent=4, sort_keys=True)
    print(nima_json)
    print("The whole json is a string!")
    print(f'Type: {type(nima_json)}')
    print(f'Length: {len(nima_json)}')

    nima_original = json.loads(nima_json)
    print(nima_original)
    print(f'Type: {type(nima_original)}')
    print(f'Length: {len(nima_original)}')
    print(f"love: {nima_original['love']}")

    print('====================================================================================')
    print('RANDOM =============================================================================')

    random.seed(1)

    print(f'10 random between 0-1: {[random.random() for _ in range(10)]}')
    print(f'10 random int from 0 including 3: {[random.randint(0, 3) for _ in range(10)]}')
    print(f'10 random int in range (0, 3): {[random.randrange(0, 3) for _ in range(10)]}')

    my_list = [10, 20, 30]
    print(f'random shuffle: {my_list}')
    random.shuffle(my_list)
    print(f'random shuffle: {my_list}')
    print(f'random selection of list: {random.choice(my_list)}')
    print(f'random selection of list 5 times: {random.choices(my_list, k=5)}')

    np.random.seed(1)
    a = np.random.random(size=(3, 3))
    print(a)
    a = np.random.randint(0, 10, size=(3, 3))
    print(a)
    np_array = np.array(my_list)
    print(np_array)
    a = np.random.choice(np_array)
    print(a)

    print('====================================================================================')
    print('DECORATOR ==========================================================================')

    print("Create a decorator which prints something before and after the function")
    def simple_decorator(function):
        def dwrapper():
            print("Something is happening before the function is called.")
            function()
            print("Something is happening after the function is called.")

        return dwrapper

    @simple_decorator
    def say_hello():
        print("Hello!")

    # Call the decorated function
    say_hello()

    def repeat(num_times):
        def decorator_repeat(func):
            def wrapper(*args, **kwargs):
                for _ in range(num_times):
                    func(*args, **kwargs)

            return wrapper

        return decorator_repeat


    @repeat(num_times=3)
    def greet(name):
        print(f"Hello, {name}!")

    # Call the decorated function
    greet("Alice")

    import time
    import functools

    def timing_decorator(func):
        # The following wrapper preserves the function attributes
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start_time = time.perf_counter()
            value = func(*args, **kwargs)
            end_time = time.perf_counter()
            run_time = end_time - start_time
            print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
            return value

        return wrapper_timer

    @timing_decorator
    def complex_calculation(num):
        result = 0
        for i in range(num):
            result += i * i
        return result

    # Call the decorated function
    value = complex_calculation(10000)
    print(value)
    print("Note that 'value' is obtained from the wrapper timer inside timing_decorator.")

    print('====================================================================================')
    print('FUNCTION ATTRIBUTES ================================================================')

    def example_function(a: int, b: int = 2) -> int:
        """This is an example function."""
        return a + b

    # Print various attributes
    print(f"Name: {example_function.__name__}")
    print(f"Docstring: {example_function.__doc__}")  # The docstring of the function, which usually describes what the function does.
    print(f"Annotations: {example_function.__annotations__}")  # A dictionary containing the function's variable annotations.
    print(f"Module: {example_function.__module__}")  # The name of the module in which the function is defined.

    print('====================================================================================')
    print('GENERATORS =========================================================================')

    print('Memory efficient and useful for large number of data.')

    # Print countdowns
    def countdown(num):
        while num > 0:
            yield num
            num -= 1

    cd = countdown(4)
    print(next(cd))
    print(next(cd))
    print(next(cd))
    print(next(cd))

    cd = countdown(4)
    for _ in range(4):
        print(next(cd))

    # Sum countdowns
    cd = countdown(4)
    print(sum(cd))

    # Create Fibonacci sequence with generators
    def fibonacci_generator():
        a, b = 0, 1
        while True:
            yield a
            a, b = b, a+b

    print('Fibonacci')
    fib_gen = fibonacci_generator()
    for _ in range(10):
        print(next(fib_gen))


    # Complex example: Generator expression for generating prime numbers
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num ** 0.5) + 1):
            if num % i == 0:
                return False
        return True

    # Generator expression for generating prime numbers
    prime_generator = (num for num in range(2, 20) if is_prime(num))

    # Using the generator to print prime numbers up to 100
    for prime in prime_generator:
        print(prime)

    print('====================================================================================')
    print('KWARGS =============================================================================')

    def my_func(a, b, c, d, e):
        print(a)
        print(b)
        print(c)
        print(d)
        print(e)

    my_dict = {'c': 1, 'd': 2, 'e': 3}
    my_func(1, 2, **my_dict)

    numbers = [1, 2, 3, 4, 5]
    start, *middle, end = numbers
    print(start)
    print(middle)
    print(end)

    your_dict = my_dict.copy()
    print(my_dict)
    print(your_dict)
    your_dict['c'] = 0
    print(my_dict)
    print(your_dict)




