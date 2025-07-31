import random as rm
import cv2
import numpy as np

def encrypt_img(img, key):
    """
    Encrypt image by shuffling pixels based on a key.
    No permutation is returned — key must be reused for decryption.
    """
    h, w, c = img.shape
    flat = img.reshape(-1, c)

    rng = np.random.default_rng(seed=key)
    shuffled = flat[rng.permutation(flat.shape[0])]
    
    return shuffled.reshape(h, w, c)

def decrypt_img(scrambled_img, key):
    """
    Decrypt image that was shuffled using the same key.
    """
    h, w, c = scrambled_img.shape
    flat = scrambled_img.reshape(-1, c)

    rng = np.random.default_rng(seed=key)
    perm = rng.permutation(flat.shape[0])
    inv_perm = np.argsort(perm)

    unshuffled = flat[inv_perm]
    return unshuffled.reshape(h, w, c)

def randKeys():
    ss1 = rm.random() * 0.02
    ss2 = rm.random() * 0.02
    ss1 = round(ss1, 5)
    ss2 = round(ss2, 5)
    return ss1, ss2

def main():
    # Example usage
    ss1, ss2 = randKeys()
    img = cv2.imread('shore_jpg.jpg')
    key = 42  # Example key

    encrypted_img = encrypt_img(img, key)
    decrypted_img = decrypt_img(encrypted_img, key)

    cv2.imwrite('encrypted_example.jpg', encrypted_img)
    cv2.imwrite('decrypted_example.jpg', decrypted_img)

if __name__ == "__main__":
    main()
