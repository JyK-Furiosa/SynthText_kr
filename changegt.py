import os


with open('word_gt.txt', 'r') as f:
    lines = f.readlines()

path = []
words = []
for line in lines:
    print(line)
    # imagePath, label = line.strip('\n').split('\t')

    # newpath = 'word_'+imagePath[4:]
    # path.append(newpath)
    # words.append(label)

# with open('word_gt.txt', 'w') as f:
#     for p, w in zip(path, words):
#         f.write(f'{p}\t{w}\n')