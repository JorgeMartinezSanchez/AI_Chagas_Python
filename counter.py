import os

def count_images_recursive(directory):
    image_extensions = ('.jpg', '.jpeg', '.png')

    count = 0
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            if file_name.lower().endswith(image_extensions):
                count += 1

    return count


if __name__ == "__main__":
    directory = input("Enter directory path: ")

    if os.path.isdir(directory):
        total_images = count_images_recursive(directory)
        print(f"Found {total_images} image(s).")
    else:
        print("Invalid directory.")