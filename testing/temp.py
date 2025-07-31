import cv2
import time
import numpy as np
import random as rm

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

def insert(img, message, ss1, ss2, numberOfBlocks):
    height, width, channel = img.shape
    img_size = height*width
    image_cap = img_size/4
    print(f'Image capacity: {image_cap} characters')                                                                            # VERBOSE
    print(len(message), " characters to embed")                                                                                 # VERBOSE
    print(f'Image - height: {height} width: {width} channel: {channel} \nimage Size: {img_size}')                               # VERBOSE
    
    opEnc = input("encrypt image (y/n): ").lower()
    if opEnc == "y":
        print("Encrypting image...")
        key = int(((ss1 + ss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(img, key)
    elif opEnc == "n":
        print("Image not encrypted")
    else:
        print("unknown command, continuing without encryption")

    # 3d to 1d matrix
    flattenImg = img.flatten()
    print('Flattened the Image from 3D to 1D')                                                                                 # VERBOSE

    # img to binary
    start = time.perf_counter()
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)
    print('Converted flattened image to binary')
    end = time.perf_counter()
    print(f'Time taken to convert image to binary: {(end - start):.3f} seconds')                                                     # VERBOSE

    # text to binary
    binary_msg = [format(ord(c), "08b") for c in message]
    msg_length = len(binary_msg)
    print("Converted text to binary")

    # assigning values for the PK component
    startPOS = int(ss1*img_size) + int(ss2*img_size)        # starting position in the image
    block_size = int(msg_length/numberOfBlocks)               # size of msg block 
    lastMsgBlock = msg_length - block_size * numberOfBlocks
    print('\nEntering insertion loop')                                                                                                   # VERBOSE
    
    for i in range(numberOfBlocks):
        if block_size == 0:
            continue
        # get the image block    
        startRange = startPOS + i * block_size * 4           # the starting range of image block
        endRange = startPOS + (i + 1) * block_size * 4       # the ending range of image block
        imgBlock_1d = binary_img[startRange:endRange]   # extracted block of 1D image array
        imgBlock_2d = np.array([list(map(int, row)) for row in imgBlock_1d]) # reshaping block from 1d to 2d

        # get the msg block
        msgStart = i * block_size                            # the starting range of message block/ similar to image 
        msgEnd = (i + 1) * block_size                        # the ending range of message block/ similar to image 
        msgBlock_1d = binary_msg[msgStart:msgEnd]       # extracted 1d msg array
        msgBlock_flat = []                              # to store values in int instead of string (below loop is to str -> int)

        for binary in msgBlock_1d:                      # each value in 1d msg array is a single value
            for bit in binary:                          # and we need to split all the values to reshape into 2D
                msgBlock_flat.append(int(bit))          # 1D msg array flattened, each bits is separated

        msgBlock_2d = np.array(msgBlock_flat).reshape(block_size*4, 2)   # 1D to 2D with 2 columns

        # inserting msg to image
        imgBlock_2d[:, 6:8] = msgBlock_2d       
        imgBlock_1d = np.array([''.join(map(str, row)) for row in imgBlock_2d])
        
        binary_img[startRange:endRange] = imgBlock_1d
    
    if lastMsgBlock > 0:
        # get the last image block
        startRange = startPOS + numberOfBlocks * block_size * 4                      # the starting range of image block
        endRange = startPOS + numberOfBlocks * block_size * 4 + lastMsgBlock * 4     # the ending range of image block

        lastImgBlock_1d = binary_img[startRange:endRange]
        lastImgBlock_2d = np.array([list(map(int, row)) for row in lastImgBlock_1d])

        # get the last message block
        msgStart = numberOfBlocks * block_size                            # the starting range of message block/ similar to image 
        msgEnd = numberOfBlocks * block_size + lastMsgBlock               # the ending range of message block/ similar to image 
        
        lastMsgBlock_1d = binary_msg[msgStart:msgEnd]

        lastMsgBlock_flat = []
        for binary in lastMsgBlock_1d:                      # each value in 1d msg array is a single value
            for bit in binary:                              # and we need to split all the values to reshape into 2D
                lastMsgBlock_flat.append(int(bit))          # 1D msg array flattened, each bits is separated

        lastMsgBlock_2d = np.array(lastMsgBlock_flat).reshape(lastMsgBlock*4, 2)   # 1D to 2D with 2 columns

        # inserting last block of message into last block of image
        lastImgBlock_2d[:, 6:8] = lastMsgBlock_2d
        lastImgBlock_1d = np.array([''.join(map(str, row)) for row in lastImgBlock_2d])

        binary_img[startRange:endRange] = lastImgBlock_1d

        start = time.perf_counter()
        pixel_values = np.array([int(pixel, 2) for pixel in binary_img], dtype=np.uint8)
        image = pixel_values.reshape((height, width, channel))

        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        end = time.perf_counter()
        print(f'Time taken to convert binary image back to pixel values: {(end - start):.3f} seconds')                                                     # VERBOSE

        if opEnc == "y":
            print("Decrypting image...")            
            image = decrypt_img(image, key)

        cv2.imwrite("stego.png", image)
        print("Image saved successfully")

        print('\t\tINSERTION COMPLETED')
        return 0

def extract(img, xss1, xss2, xnumberOfBlocks, msg_length):    
    height, width, channel = img.shape
    img_size = height*width*channel
    print(f'Image - height: {height} width: {width} channel: {channel} \nimage Size: {img_size}')                               # VERBOSE

    opEnc = input("encrypt image before extraction (y/n): ").lower()
    if opEnc == "y":
        print("Encrypting image...")
        key = int(((xss1 + xss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(img, key)
    elif opEnc == "n":
        print("Image not encrypted")
    else:
        print("unknown command, continuing without encryption")


    # 3d to 1d matrix
    flattenImg = img.flatten()
    print('Flattened the Image from 3D to 1D')                                                                                 # VERBOSE                                                                                     # VERBOSE

    # img to binary
    start = time.perf_counter()
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)
    print('Converted flattened image to binary')
    end = time.perf_counter()
    print(f'Time taken to convert image to binary: {(end - start):.3f} seconds')                                                     # VERBOSE

    xstartPOS = int(xss1*img_size) + int(xss2*img_size)        # starting position in the image
    xblock_size = int(msg_length/xnumberOfBlocks)               # size of msg block 
    xlastMsgBlock = msg_length - xblock_size * xnumberOfBlocks

    print('\nEntering extraction loop')                                                                                                      # VERBOSE
    dmes = []
    for i in range(xnumberOfBlocks):
        # get the image block
        startRange = xstartPOS + i * xblock_size * 4           # the starting range of image block
        endRange = xstartPOS + (i + 1) * xblock_size * 4       # the ending range of image block
        imgBlock_1d = binary_img[startRange:endRange]   # extracted block of 1D image array
        imgBlock_2d = np.array([list(map(int, row)) for row in imgBlock_1d]) # reshaping block from 1d to 2d

        # extract the msg from 2LSB
        # Extract last two bits 
        msgBit = [[str(bit) for bit in bits[-2:]] for bits in imgBlock_2d]
        
        # Reshape to P1 rows x 8 columns
        msgBin = np.reshape(msgBit, (xblock_size, 8))
        #print("extracted binary message >>>>> \n", msgBin)
        # Convert binary strings to decimal
        msgBlock = [int(''.join(bits), 2) for bits in msgBin]
        dmes.extend(msgBlock)
    
    # Process the last block if there is one
    if xlastMsgBlock > 0:
        startRange = xstartPOS + xnumberOfBlocks * xblock_size * 4                      # the starting range of image block
        endRange = xstartPOS + xnumberOfBlocks * xblock_size * 4 + xlastMsgBlock * 4     # the ending range of image block
        lastImgBlock_1d = binary_img[startRange:endRange]
        lastImgBlock_2d = np.array([list(map(int, row)) for row in lastImgBlock_1d])

        # Extract last two bits
        msgBit = [[str(bit) for bit in bits[-2:]] for bits in lastImgBlock_2d]
        # Reshape to LST1 rows x 8 columns
        msgBin = np.reshape(msgBit, (xlastMsgBlock, 8))
        # Convert binary strings to decimal
        msgBlock = [int(''.join(bits), 2) for bits in msgBin]
        dmes.extend(msgBlock)

    # Convert decimal values to characters
    Omessage = ''.join([chr(c) for c in dmes])

    print('\t\tExtraction COMPLETED')
    return Omessage

def randKeys():
    ss1 = rm.random() * 0.02
    ss2 = rm.random() * 0.02
    ss1 = round(ss1, 5)
    ss2 = round(ss2, 5)
    return ss1, ss2

def main():
    option = 88
    while option != 0:
        print("""
    1. Embed msg in image
    2. Extract msg from image
    0. EXIT          
    """)
        option = int(input("select option: "))
        if option == 1:
            print("option 1 selected")
            img_path = "shore_jpg.jpg"# input("image name: ")
            img = cv2.imread(img_path)  # Reading an image
            message = "yo yo yoooo it's your devy boyyy - viZzyyh"#input("message to embed: ")
            msg_length = len(message)
            print(msg_length)
            opKey = input("generate keys (y/n): ").lower()
            if opKey == "y":
                print("\nSAVE THESE KEYS!")
                ss1, ss2 = randKeys()
                print("ss1: ", ss1)
                print("ss2: ", ss2)
            elif opKey == "n":
                ss1 = float(input("enter key value 01: "))     # scaling step 1
                ss2 = float(input("enter key value 02: "))    # scaling step 2
            else:
                print("unknown command")
                break
            
            numberOfBlocks = 4
            start = time.perf_counter()
            insert(img, message, ss1, ss2, numberOfBlocks)
            end = time.perf_counter()
            print(f"Time taken to insert message: {(end - start):.3f} seconds")

        elif option == 2:
            print("option 2 selected")
            img_path = "stego.png"# input("image name: ")
            img = cv2.imread(img_path)  # Reading an image
            xss1 = float(input("enter key value 01: "))     # scaling step 1
            xss2 = float(input("enter key value 02: "))    # scaling step 2
            xnumberOfBlocks = 4
            msg_length = 42
            start = time.perf_counter()
            print("Extracted message: ", extract(img, xss1, xss2, xnumberOfBlocks, msg_length))
            end = time.perf_counter()
            print(f"Time taken to extract message: {(end - start):.3f} seconds")

        else: 
            print("option 0 selected")
            continue

# This ensures main() runs only when this script is executed directly
if __name__ == "__main__":
    main()

# added dynamic key during insertion and requires key to extract
# saves image as a png for perfect extraction