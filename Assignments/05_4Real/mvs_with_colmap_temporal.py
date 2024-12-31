import os
import shutil
from argparse import ArgumentParser
import cv2


def get_register_image(freeze_data_dir: str, animated_data_dir: str):
    print('Selecting register image...')
    animated_images = {}
    for animated_image_name in os.listdir(os.path.join(animated_data_dir, 'images')):
        if len(animated_image_name) >= 4 and animated_image_name[-4:] == '.png':
            animated_image = cv2.imread(os.path.join(animated_data_dir, 'images', animated_image_name))
            animated_images[animated_image_name] = animated_image
    freeze_images = {}
    for freeze_image_name in os.listdir(os.path.join(freeze_data_dir, 'images')):
        if len(freeze_image_name) >= 4 and freeze_image_name[-4:] == '.png':
            freeze_image = cv2.imread(os.path.join(freeze_data_dir, 'images', freeze_image_name))
            freeze_images[freeze_image_name] = freeze_image
    min_values = None
    for animated_image_name, animated_image in animated_images.items():
        print(animated_image_name)
        for freeze_image_name, freeze_image in freeze_images.items():
            value = (animated_image - freeze_image).mean() + (freeze_image - animated_image).mean()
            if min_values is None or min_values[0] > value:
                min_values = [value, animated_image_name, freeze_image_name]
    print(min_values)
    return min_values[2]

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate COLMAP data in animated video image dataset')
    parser.add_argument('--freeze_data_dir', type=str, required=True, help='Path to the freeze-time video input directory containing images in data_dir/images')
    parser.add_argument('--animated_data_dir', type=str, required=True, help='Path to the animated video input directory containing images in data_dir/images')
    parser.add_argument('--register_image', type=str, default=None, help='Register animated video into the registered image in freeze-time video')
    args = parser.parse_args()

    if args.register_image is None:
        args.register_image = get_register_image(args.freeze_data_dir, args.animated_data_dir)
    elif not os.path.exists(os.path.join(args.freeze_data_dir, 'images', args.register_image)):
        raise RuntimeError(f'Cannot find register image {args.register_image}')

    shutil.rmtree(os.path.join(args.animated_data_dir, 'sparse'), ignore_errors=True)
    shutil.rmtree(os.path.join(args.animated_data_dir, 'colmap'), ignore_errors=True)
    shutil.copytree(os.path.join(args.freeze_data_dir, 'sparse'), os.path.join(args.animated_data_dir, 'sparse'))
    shutil.rmtree(os.path.join(args.animated_data_dir, 'sparse', '0'))
    os.rename(os.path.join(args.animated_data_dir, 'sparse', '0_text'), os.path.join(args.animated_data_dir, 'sparse', '0'))

    images_content = ''
    should_read_points2D = False
    should_stop = False
    with open(os.path.join(args.animated_data_dir, 'sparse', '0', 'images.txt')) as f:
        for line in f:
            if len(line) == 0 or line[0] == '#':
                images_content += line
                continue
            if not should_read_points2D:
                cell = line.split()
                image_name = cell[9]
                if image_name == args.register_image:
                    image_data = line
                    should_stop = True
                should_read_points2D = True
            else:
                should_read_points2D = False
                if should_stop:
                    points2D = line
                    break
    cell[0] = '{}'
    cell[9] = '{}'
    image_data_pattern = ' '.join(cell)
    with open(os.path.join(args.animated_data_dir, 'sparse', '0', 'images.txt'), 'w') as f:
        f.write(images_content)
        image_files = [file for file in os.listdir(os.path.join(args.animated_data_dir, 'images')) if len(file) >= 4 and file[-4:] == '.png']
        image_idx = 1
        for image_name in sorted(image_files, key=lambda x: int(x[2:-4])):
            f.write(image_data_pattern.format(image_idx, image_name) + '\n')
            image_idx += 1
            f.write(points2D)

    os.makedirs(os.path.join(args.animated_data_dir, 'colmap', 'stereo'), exist_ok=True)
    shutil.copytree(os.path.join(args.animated_data_dir, 'sparse', '0'), os.path.join(args.animated_data_dir, 'colmap', 'stereo', 'sparse'))
