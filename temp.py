import random as rm

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



def main():
    text_path = "test.txt"
    with open(text_path, "r", encoding="utf-8") as f:
        message = f.read()
    msg_length = len(message)
    print(f"Message length: {msg_length}")

    ss1, ss2, seed = randKeys(msg_length)
    print("ss1: ", ss1)
    print("ss2: ", ss2)
    print("seed: ", seed)

    ss1_str = f"{ss1:.5f}"
    ss2_str = f"{ss2:.5f}"
    seed_int = seed
    print("ss1 (str):", ss1_str)
    print("ss2 (str):", ss2_str)
    print("seed (int):", seed_int)
    
    msg_length1 = length_from_keys(ss1, ss2, seed)
    print(f"Reconstructed message length: {msg_length1}")

if __name__ == "__main__":
    main()