import os

root_dir = 'data/flowers/'

f = open('meta/train.txt', 'w')
classes = os.listdir('train')
classes = sorted(classes)
for c in classes:
    print(c)
    class_id = int(c.split("_")[-1])
    cur_files = os.listdir(os.path.join('train', c))
    cur_files = sorted(cur_files)
    for cur_file in cur_files:
        line = root_dir + 'train/' + c + '/' + cur_file + ' ' + str(class_id) + '\n'
        f.write(line)
f.close()

f = open('meta/val.txt', 'w')
classes = os.listdir('val')
classes = sorted(classes)
for c in classes:
    print(c)
    class_id = int(c.split("_")[-1])
    cur_files = os.listdir(os.path.join('val', c))
    cur_files = sorted(cur_files)
    for cur_file in cur_files:
        line = root_dir + 'val/' + c + '/' + cur_file + ' ' + str(class_id) + '\n'
        f.write(line)
f.close()

f = open('meta/test.txt', 'w')
classes = os.listdir('test')
classes = sorted(classes)
for c in classes:
    print(c)
    class_id = int(c.split("_")[-1])
    cur_files = os.listdir(os.path.join('test', c))
    cur_files = sorted(cur_files)
    for cur_file in cur_files:
        line = root_dir + 'test/' + c + '/' + cur_file + ' ' + str(class_id) + '\n'
        f.write(line)
f.close()
