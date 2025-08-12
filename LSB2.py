import cv2
import time
import numpy as np

DELIMITER = "####"

def insert(img, message, verbose=True):

    if verbose: print("[+] Flattening image...")
    flat = img.flatten()

    if verbose: print("[+] Converting image to binary...")
    binary_flat = np.vectorize(lambda x: format(x, '08b'))(flat)

    full_message = message + DELIMITER
    binary_msg = ''.join(format(ord(c), '08b') for c in full_message)

    if len(binary_msg) > len(binary_flat) * 2:
        raise ValueError("Message too large for this image.")

    if verbose: print("[+] Embedding message...")
    start = time.perf_counter()
    idx = 0
    modified_pixels = []
    for pixel_bin in binary_flat:
        if idx + 2 <= len(binary_msg):
            new_pixel = pixel_bin[:-2] + binary_msg[idx:idx+2]
            idx += 2
        else:
            new_pixel = pixel_bin
        modified_pixels.append(new_pixel)
    end = time.perf_counter()
    print(f"[+] LSB2 Time taken to embed message:     {(end - start):.5f} seconds")                                             # VERBOSE
    
    modified_flat = np.array([int(b, 2) for b in modified_pixels], dtype=np.uint8)
    stego_img = modified_flat.reshape(img.shape)
    return stego_img

def extract(img, verbose=True):
    if img is None:
        raise ValueError("Image is None.")

    if verbose: print("[+] Flattening image...")
    flat = img.flatten()

    if verbose: print("[+] Converting image to binary...")
    binary_flat = np.vectorize(lambda x: format(x, '08b'))(flat)

    if verbose: print("[+] Extracting message...")
    start = time.perf_counter()
    bits = ''.join(b[-2:] for b in binary_flat)
    chars = [bits[i:i+8] for i in range(0, len(bits), 8)]

    message = ''
    for c in chars:
        message += chr(int(c, 2))
        if message.endswith(DELIMITER):
            end = time.perf_counter()
            print(f"[+] LSB2 Time taken to extract message:   {(end - start):.5f} seconds")                                             # VERBOSE
            return message[:-len(DELIMITER)]
    end = time.perf_counter()
    print(f"[+] LSB2 Time taken to extract message:   {(end - start):.5f} seconds")                                             # VERBOSE
    return "[!] No valid message found."

def main():
    mode = input("Mode (e=embed / d=decode): ").strip().lower()
    if mode == 'e':
        cover_path = "shore_jpg.jpg"         #input("Enter path to cover JPG image: ")
        message = "secret text yoo"          #input("Enter message to hide: ")
        img = cv2.imread(cover_path)
        stego = insert(img, message, verbose=True)
        cv2.imwrite("stegoLSB2.png", stego)
        print("[+] Message embedded and saved as stegoLSB2.png")
    elif mode == 'd':
        stego_path = "stegoLSB2.png"         #input("Enter path to stego PNG image: ")
        img = cv2.imread(stego_path)
        result = extract(img, verbose=True)
        print("[+] Extracted message:", result)
    else:
        print("[!] Invalid mode.")

if __name__ == "__main__":
    main()
