import cv2
import numpy as np
import time

DELIMITER = "####"

def insert(cover_path, message, output_path="stego_output.png"):
    start = time.perf_counter()
    print("reading image...")
    img = cv2.imread(cover_path)
    if img is None:
        raise ValueError("Image not found or unreadable.")

    # Flatten the 3D image array into 1D
    print("[+] Flattening image...")
    flat = img.flatten()

    # Convert each pixel value to an 8-bit binary string
    print("[+] Converting image to binary...")
    start1 = time.perf_counter()
    binary_flat = np.vectorize(lambda x: format(x, '08b'))(flat)
    end1 = time.perf_counter()
    print(f"[+] Time taken to convert image to binary: {end1 - start1:.5f} seconds")

    # Append delimiter to message and convert message to binary string
    full_message = message + DELIMITER
    binary_msg = ''.join(format(ord(c), '08b') for c in full_message)

    # Check if the message can fit in the image (2 bits per pixel byte)
    if len(binary_msg) > len(binary_flat) * 2:
        raise ValueError("Message too large for this image.")

    # Embedding message into image
    start2 = time.perf_counter()
    idx = 0
    modified_pixels = []
    for pixel_bin in binary_flat:
        if idx + 2 <= len(binary_msg):
            new_pixel = pixel_bin[:-2] + binary_msg[idx:idx+2]
            idx += 2
        else:
            new_pixel = pixel_bin
        modified_pixels.append(new_pixel)
    end2 = time.perf_counter()
    print(f"[+] Time taken to embed message: {end2 - start2:.5f} seconds")

    # Convert modified binary values back to integers
    modified_flat = np.array([int(b, 2) for b in modified_pixels], dtype=np.uint8)

    # Reshape 1D pixel array back to original image shape
    stego_img = modified_flat.reshape(img.shape)
    cv2.imwrite("stegoLSB2.png", stego_img)
    print(f"[+] Message embedded and saved")
    end = time.perf_counter()
    print(f"[+] Total time taken: {end - start:.5f} seconds")

def extract(stego_path):
    img = cv2.imread(stego_path)
    if img is None:
        raise ValueError("Image not found or unreadable.")

    # Flatten the image to 1D
    print("[+] Flattening image...")
    flat = img.flatten()

    # Convert each pixel to binary
    print("[+] Converting image to binary...")
    start1 = time.perf_counter()
    binary_flat = np.vectorize(lambda x: format(x, '08b'))(flat)
    end1 = time.perf_counter()
    print(f"[+] Time taken to convert image to binary: {end1 - start1:.5f} seconds")

    print("[+] Extracting message...")
    start2 = time.perf_counter()
    # Extract 2 LSBs from each byte and combine into a long bitstring
    bits = ''.join(b[-2:] for b in binary_flat)
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]

    # Convert bits to characters until the delimiter is found
    message = ''
    for c in chars:
        message += chr(int(c, 2))
        if message.endswith(DELIMITER):
            end2 = time.perf_counter()
            print(f"[+] Time taken to extract message: {end2 - start2:.5f} seconds")
            return message[:-len(DELIMITER)]
    end2 = time.perf_counter()
    print(f"[+] Time taken to extract message: {end2 - start2:.5f} seconds")     
    return "[!] No valid message found."

def main():
    mode = input("Mode (e=embed / d=decode): ").strip().lower()
    if mode == 'e':
        cover = "shore_jpg.jpg"         #input("Enter path to cover JPG image: ")
        msg = "secret text yoo"         #input("Enter message to hide: ")
        insert(cover, msg)
    elif mode == 'd':
        stego = "stegoLSB2.png"         #input("Enter path to stego PNG image: ")
        result = extract(stego)
        print("[+] Extracted message:", result)
    else:
        print("[!] Invalid mode.")

if __name__ == "__main__":
    main()
