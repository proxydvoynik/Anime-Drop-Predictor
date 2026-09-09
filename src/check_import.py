import train_split, inspect;
print(train_split.__file__);
print([n for n in dir(train_split) if not n.startswith('_')])