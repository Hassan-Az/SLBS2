import cv2
import time
import numpy as np
import random as rm

def encrypt_img(img, key):
    """Encrypt image by shuffling pixels based on a key."""
    h, w, c = img.shape
    flat = img.reshape(-1, c)

    rng = np.random.default_rng(seed=key)
    shuffled = flat[rng.permutation(flat.shape[0])]
    
    return shuffled.reshape(h, w, c)

def decrypt_img(scrambled_img, key):
    """Decrypt image that was shuffled using the same key."""
    h, w, c = scrambled_img.shape
    flat = scrambled_img.reshape(-1, c)

    rng = np.random.default_rng(seed=key)
    perm = rng.permutation(flat.shape[0])
    inv_perm = np.argsort(perm)

    unshuffled = flat[inv_perm]
    return unshuffled.reshape(h, w, c)

def insert(image, message, ss1, ss2, numberOfBlocks):
    height, width, channel = image.shape
    img_size = height*width
    
    opEnc = input("\n> Encrypt image (y/n): ").lower()
    if opEnc == "y":
        key = int(((ss1 + ss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(image, key)
        print("+ Image encrypted")
    elif opEnc == "n":
        img = image
        print("+ Encryption skipped\n")

    start1 = time.perf_counter()

    # 3d to 1d matrix
    print('+ processing - Flattening the Image from 3D to 1D', end="", flush=True)                                                                                 # VERBOSE
    flattenImg = img.flatten()
    time.sleep(1)  # Adding a small delay for better readability in output

    # img to binary
    print('\r+ processing - Converting flattened image to binary', end="", flush=True)                                 # VERBOSE
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)
    time.sleep(1)  # Adding a small delay for better readability in output

    # text to binary
    print("\r+ processing - Converting text to binary           ", end="", flush=True)                                  # VERBOSE
    binary_msg = [format(ord(c), "08b") for c in message]
    msg_length = len(binary_msg)
    time.sleep(1)  # Adding a small delay for better readability in output

    print("\r+ processing - Completed                            ", end="", flush=True)                                 # VERBOSE
    time.sleep(1)  # Adding a small delay for better readability in output

    # assigning values for the PK component
    startPOS = int(ss1*img_size) + int(ss2*img_size)        # starting position in the image
    block_size = int(msg_length/numberOfBlocks)               # size of msg block 
    lastMsgBlock = msg_length - block_size * numberOfBlocks
    
    print("\n+ Embedding message...")
    start = time.perf_counter()
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
        end = time.perf_counter()
        print(f"[+] Time taken to insert message: {(end - start):.5f} seconds")                                                                 # VERBOSE
        print("+ Message embedded successfully")   

        # Converting binary image to image
        print('\n+ processing - Converting binary image to 3d image ', end="", flush=True)
        pixel_values = np.array([int(pixel, 2) for pixel in binary_img], dtype=np.uint8)
        image = pixel_values.reshape((height, width, channel))    

        if opEnc == "y":
            print("\r+ processing - Saving Decrypted Image               ", end="", flush=True)                                                                                                       # VERBOSE
            image = decrypt_img(image, key)            

        print("\r+ processing - Completed                            ", end="", flush=True)                                 # VERBOSE

        cv2.imwrite("stego.png", image)
        end1 = time.perf_counter()
        print(f"\n[+] Total time taken: {(end1 - start1):.5f} seconds")                                                                                                       # VERBOSE
        return 0

def extract(image, xss1, xss2, xnumberOfBlocks, msg_length):    
    height, width, channel = image.shape
    img_size = height*width

    opEnc = input("\n> Encrypt image (y/n): ").lower()
    if opEnc == "y":
        key = int(((xss1 + xss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(image, key)
        print("+ Image encrypted")
    else:
        img = image

    start = time.perf_counter()

    # 3d to 1d matrix
    print('+ processing - Flattening the Image from 3D to 1D', end="", flush=True)                                                                                 # VERBOSE
    flattenImg = img.flatten()
    time.sleep(1)  # Adding a small delay for better readability in output

    # img to binary
    print('\r+ processing - Converting flattened image to binary', end="", flush=True)
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)
    time.sleep(1)  # Adding a small delay for better readability in output

    print("\r+ processing - Completed                            ", end="", flush=True)
    time.sleep(1)  # Adding a small delay for better readability in output

    xstartPOS = int(xss1*img_size) + int(xss2*img_size)        # starting position in the image
    xblock_size = int(msg_length/xnumberOfBlocks)               # size of msg block 
    xlastMsgBlock = msg_length - xblock_size * xnumberOfBlocks

    print("\n+ Extracting message...")
    start = time.perf_counter()
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

    print('\n+ Extracted successfully')
    end = time.perf_counter()
    print(f"[+] Time taken to extract message: {(end - start):.5f} seconds")
    return Omessage

def randKeys():
    ss1 = rm.random() * 0.02
    ss2 = rm.random() * 0.02
    ss1 = round(ss1, 5)
    ss2 = round(ss2, 5)
    return ss1, ss2

def imgcap(img):
    height, width, channel = img.shape
    img_size = height*width
    image_cap = img_size/4
    print(f'\n+ Image capacity: {image_cap} characters')

def main():
    option = 88
    while option != 0:
        print("""
    1. Embed msg in image
    2. Extract msg from image
    3. Show image capacity
    0. EXIT          
    """)
        option = int(input("> select option: "))
        if option == 1:
            print("+ option 1 selected")
            img_path = "shore_jpg.jpg"# input("image name: ")
            img = cv2.imread(img_path)  # Reading an image
            message = "yo yo yoooo it's your devy boyyy - viZzyyh"#input("message to embed: ")
            opKey = input("\n> generate keys (y/n): ").lower()
            if opKey == "y":
                print("____________________")
                print("SAVE THESE KEYS!")
                ss1, ss2 = randKeys()
                print("ss1: ", ss1)
                print("ss2: ", ss2)
                print("____________________")
            elif opKey == "n":
                ss1 = float(input("\n> enter key value 01: "))     # scaling step 1
                ss2 = float(input("\n> enter key value 02: "))    # scaling step 2
            else:
                print("unknown command")
                break
            
            numberOfBlocks = 4
            
            insert(img, message, ss1, ss2, numberOfBlocks)

        elif option == 2:
            print("+ option 2 selected")
            img_path = "stego.png"# input("image name: ")
            img = cv2.imread(img_path)  # Reading an image
            xss1 = float(input("\n> enter key value 01: "))     # scaling step 1
            xss2 = float(input("> enter key value 02: "))    # scaling step 2
            xnumberOfBlocks = 4
            msg_length = 42            
            print("\n+ Extracted message: ", extract(img, xss1, xss2, xnumberOfBlocks, msg_length))
        
        elif option == 3:
            print("+ option 3 selected")
            img_path = "shore_jpg.jpg"
            img = cv2.imread(img_path)  # Reading an image
            imgcap(img)
        else: 
            print("+ option 0 selected")
            continue

# This ensures main() runs only when this script is executed directly
if __name__ == "__main__":
    main()

# added dynamic key during insertion and requires key to extract
# saves image as a png for perfect extraction