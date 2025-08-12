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

def insert(image, message, ss1, ss2, numberOfBlocks, verbose, opEnc="n"):
    height, width, channel = image.shape
    img_size = height*width
        
    if opEnc == "y":
        key = int(((ss1 + ss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(image, key)
        if verbose:
            print("+ Image encrypted")
    elif opEnc == "n":
        img = image
        if verbose:
            print("+ Encryption skipped\n")

    # 3d to 1d matrix
    if verbose:
        print('+ processing - Flattening the Image from 3D to 1D', end="", flush=True)                                      # VERBOSE
        time.sleep(1)  # Adding a small delay for better readability in output
    flattenImg = img.flatten()

    # img to binary
    if verbose:
        print('\r+ processing - Converting flattened image to binary', end="", flush=True)                                 # VERBOSE
        time.sleep(1)  # Adding a small delay for better readability in output
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)

    # text to binary
    if verbose:
        print("\r+ processing - Converting text to binary           ", end="", flush=True)                                  # VERBOSE
        time.sleep(1)  # Adding a small delay for better readability in output
    binary_msg = [format(ord(c), "08b") for c in message]
    msg_length = len(binary_msg)

    if verbose:
        print("\r+ processing - Completed                            ", end="", flush=True)                                 # VERBOSE
        time.sleep(1)  # Adding a small delay for better readability in output

    # assigning values for the PK component
    startPOS = int(ss1*img_size) + int(ss2*img_size)        # starting position in the image
    block_size = int(msg_length/numberOfBlocks)               # size of msg block 
    lastMsgBlock = msg_length - block_size * numberOfBlocks        

    if verbose:
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
    print(f"[+] SLSB2 Time taken to embed message:    {(end - start):.5f} seconds")                                             # VERBOSE
    if verbose:
        print("+ Message embedded successfully")   
    # Converting binary image to image
    if verbose:
        print('\n+ processing - Converting binary image to 3d image ', end="", flush=True)
    pixel_values = np.array([int(pixel, 2) for pixel in binary_img], dtype=np.uint8)
    image = pixel_values.reshape((height, width, channel))    
    if opEnc == "y":
        if verbose:            
            print("\r+ processing - Saving Decrypted Image               ", end="", flush=True)                             # VERBOSE
        image = decrypt_img(image, key)            
    if verbose:
        print("\r+ processing - Completed                            ", end="", flush=True)                                 # VERBOSE
    return image

def extract(image, xss1, xss2, xnumberOfBlocks, msg_length, verbose, opEnc="n"):    
    height, width, channel = image.shape
    img_size = height*width

    if opEnc == "y":
        key = int(((xss1 + xss2)/0.04)*255)                    # generate key from ss1 and ss2 that ranges from 0 to 255
        img = encrypt_img(image, key)
        if verbose:
            print("+ Image encrypted")
    else:
        img = image

    # 3d to 1d matrix
    if verbose:
        print('+ processing - Flattening the Image from 3D to 1D', end="", flush=True)                                                                                 # VERBOSE
        time.sleep(1)  # Adding a small delay for better readability in output
    flattenImg = img.flatten()

    # img to binary
    if verbose:
        print('\r+ processing - Converting flattened image to binary', end="", flush=True)
        time.sleep(1)  # Adding a small delay for better readability in output
    binary_img = np.vectorize(lambda x: format(x, '08b'))(flattenImg)

    if verbose:
        print("\r+ processing - Completed                            ", end="", flush=True)
        time.sleep(1)  # Adding a small delay for better readability in output

    xstartPOS = int(xss1*img_size) + int(xss2*img_size)        # starting position in the image
    xblock_size = int(msg_length/xnumberOfBlocks)               # size of msg block 
    xlastMsgBlock = msg_length - xblock_size * xnumberOfBlocks    

    if verbose:
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
    end = time.perf_counter()
    print(f"[+] SLSB2 Time taken to extract message:  {(end - start):.5f} seconds")                                             # VERBOSE
    # Convert decimal values to characters
    Omessage = ''.join([chr(c) for c in dmes])
    if verbose:
        print('\n+ Extracted successfully')
    return Omessage

def randKeys(msg_length):
    """
    Encode msg_length into ss1, ss2 (range 0..0.15) and return (ss1, ss2, seed).
    The seed must be kept/sent to the extractor so it can decode exactly.
    """

    BASE = 100000                # 5 decimal places
    M_UNITS = int(0.15 * BASE)   # = 15000

    # capacity check: requires high < M_UNITS
    low  = msg_length % M_UNITS
    high = msg_length // M_UNITS
    if high >= M_UNITS:
        raise ValueError(f"msg_length too large; max supported is {M_UNITS * M_UNITS - 1}")

    # random seed (returned so extractor can decode)
    seed = rm.randint(0, 2**31 - 1)
    rng = rm.Random(seed)

    # random offsets (same range as units)
    r1 = rng.randrange(M_UNITS)
    r2 = rng.randrange(M_UNITS)

    # embed (modular)
    ss1_units = (low  + r1) % M_UNITS
    ss2_units = (high + r2) % M_UNITS

    # convert to float keys in range [0, 0.14999...] and round to 5 decimals
    key1 = round(ss1_units / BASE, 5)
    key2 = round(ss2_units / BASE, 5)

    return key1, key2, seed


def length_from_keys(key1, key2, seed):
    """
    Recover original msg_length from (key1, key2, seed).
    Returns msg_length (single variable).
    """

    BASE = 100000                # 5 decimal places
    M_UNITS = int(0.15 * BASE)   # = 15000
    
    rng = rm.Random(seed)
    r1 = rng.randrange(M_UNITS)
    r2 = rng.randrange(M_UNITS)

    # convert back to integer units
    ss1_units = int(round(key1 * BASE))
    ss2_units = int(round(key2 * BASE))

    # undo the modular shift
    low  = (ss1_units - r1) % M_UNITS
    high = (ss2_units - r2) % M_UNITS

    msg_length = high * M_UNITS + low
    return msg_length


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
            img_path = "cover_img/img1.jpg"# input("image name: ")
            text_path = "test.txt"
            img = cv2.imread(img_path)  # Reading an image
            with open(text_path, "r", encoding="utf-8") as f:
                message = f.read()
            msg_length = len(message)            
            opKey = input("\n> generate keys (y/n): ").lower()
            if opKey == "y":
                print("____________________")
                print("SAVE THESE KEYS!")
                ss1, ss2, seed = randKeys(msg_length)                        
                print("ss1:", ss1)
                print("ss2:", ss2)
                print("seed:", seed)
                print("____________________")
            elif opKey == "n":
                ss1 = float(input("\n> enter key value 01: "))      # scaling step 1
                ss2 = float(input("\n> enter key value 02: "))      # scaling step 2
                seed = int(input("\n> enter seed value: "))    
            else:
                print("unknown command")
                break
            opEnc = input("\n> Encrypt image (y/n): ").lower()        
            numberOfBlocks = 4            
            stego_img = insert(img, message, ss1, ss2, numberOfBlocks, verbose=True, opEnc=opEnc)
            cv2.imwrite("stego.png", stego_img)

        elif option == 2:
            print("+ option 2 selected")
            img_path = "stego.png"# input("image name: ")
            img = cv2.imread(img_path)  # Reading an image
            xss1 = float(input("\n> enter key value 01: "))     # scaling step 1
            xss2 = float(input("> enter key value 02: "))    # scaling step 2
            seed = int(input("> enter seed: "))
            msg_length = length_from_keys(xss1, xss2, seed)
            xnumberOfBlocks = 4
            opEnc = input("\n> Encrypt image (y/n): ").lower()
            content = extract(img, xss1, xss2, xnumberOfBlocks, msg_length, verbose=True, opEnc=opEnc)
            with open("stego_out.txt", "w", encoding="utf-8") as f:
                f.write(content)
            print("\n+ Extracted message: ", content)

        elif option == 3:
            print("+ option 3 selected")
            img_path = "cover_img/img1.jpg"
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