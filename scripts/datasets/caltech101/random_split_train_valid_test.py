import argparse
import os
import glob
import random

def random_split(images):
    random.shuffle(images)
    size = len(images)
    i1  = int(size * 0.1)
    i2  = int(size * 0.2)

    train_images = images[i2:]
    valid_images = images[i1:i2]
    test_images  = images[0:i1]

    return (train_images, valid_images, test_images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--out', default='.')
    parser.add_argument('--seed', default=0)
    args = parser.parse_args()

    dir = args.dir
    labels = sorted(os.listdir(dir))
    random.seed(args.seed)

    outdir = os.path.abspath(args.out)
    train_path = os.path.join(outdir, "train.txt")
    valid_path = os.path.join(outdir, "valid.txt")
    test_path  = os.path.join(outdir, "text.txt")
    label_path = os.path.join(outdir, "label.txt")

    with open(train_path, 'w') as train_f, open(valid_path, 'w') as valid_f, open(test_path, 'w') as test_f, open(label_path, 'w') as label_f:
        for i, label in enumerate(labels):
            images = glob.glob(os.path.join(dir, label, '*'))
            (train_images, valid_images, test_images) = random_split(images)

            for path in train_images:
                train_f.write("{} {}\n".format(path, i))
            for path in valid_images:
                valid_f.write("{} {}\n".format(path, i))
            for path in test_images:
                test_f.write("{} {}\n".format(path, i))
            label_f.write("%d %s\n" % (i, label))

if __name__ == '__main__':
    main()
