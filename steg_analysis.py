import os
import cv2
import time
import numpy as np
from LSB2 import insert as lsb2_insert, extract as lsb2_extract
from SLSB2 import insert as silsb2_insert, extract as silsb2_extract, length_from_keys, randKeys

# Fixed parameters
numberOfBlocks = 4
with open("test.txt", "r", encoding="utf-8") as f:
    message = f.read()
ss1, ss2, seed = randKeys(len(message))
output_dir = "output"

os.makedirs(output_dir, exist_ok=True)

# === Helper functions ===
def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def psnr(img1, img2):
    error = mse(img1, img2)
    if error == 0:
        return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(error))



def diff_image(original, stego, path):    
    diff = cv2.absdiff(original, stego)
    # Optional: Enhance visibility of small differences
    diff_enhanced = cv2.convertScaleAbs(diff, alpha=10)  # Scale the difference for better visibility
    # Save the diff image directly (no grayscale conversion)
    cv2.imwrite(path, diff_enhanced)


def show_side_by_side(original, lsb2, silsb2, path):
    stacked = np.hstack((original, lsb2, silsb2))
    cv2.imwrite(path, stacked)

# === Main Analysis ===
def main():
    #print("[+] Enter path to original image:")
    #path = input("> ").strip()
    cover = cv2.imread("cover_img/img1.jpg")
    if cover is None:
        print("[!] Could not read image.")
        return

    print("[+] Starting analysis...\n")

    # -- LSB2 --
    start = time.perf_counter()
    stego_lsb2 = lsb2_insert(cover.copy(), message, verbose=False)
    embed_time_lsb2 = time.perf_counter() - start
    print(f"[+] LSB2 embedding total time:            {embed_time_lsb2:.5f} seconds")

    start1 = time.perf_counter()
    extracted_lsb2 = lsb2_extract(stego_lsb2, verbose=False)
    extract_time_lsb2 = time.perf_counter() - start1
    print(f"[+] LSB2 extraction total time:           {extract_time_lsb2:.5f} seconds\n")

    # -- SILSB2 --
    start2 = time.perf_counter()
    stego_silsb2 = silsb2_insert(cover.copy(), message, ss1, ss2, numberOfBlocks, verbose=False, opEnc="y")
    embed_time_silsb2 = time.perf_counter() - start2
    print(f"[+] SILSB2 embedding total time:          {embed_time_silsb2:.5f} seconds")

    start3 = time.perf_counter()
    msg_length = length_from_keys(ss1, ss2, seed)
    extracted_silsb2 = silsb2_extract(stego_silsb2, ss1, ss2, numberOfBlocks, msg_length, verbose=False, opEnc="y")
    extract_time_silsb2 = time.perf_counter() - start3
    print(f"[+] SILSB2 extraction total time:         {extract_time_silsb2:.5f} seconds")

    # -- Save stego images --
    path_lsb2 = os.path.join(output_dir, "stego_lsb2.png")
    path_silsb2 = os.path.join(output_dir, "stego_silsb2.png")
    cv2.imwrite(path_lsb2, stego_lsb2)
    cv2.imwrite(path_silsb2, stego_silsb2)

    # -- Metrics --
    mse_lsb2 = mse(cover, stego_lsb2)
    mse_silsb2 = mse(cover, stego_silsb2)

    psnr_lsb2 = psnr(cover, stego_lsb2)
    psnr_silsb2 = psnr(cover, stego_silsb2)

    # -- Save visual comparisons --
    diff_image(cover, stego_lsb2, os.path.join(output_dir, "diff_lsb2.png"))
    diff_image(cover, stego_silsb2, os.path.join(output_dir, "diff_silsb2.png"))
    show_side_by_side(cover, stego_lsb2, stego_silsb2, os.path.join(output_dir, "side_by_side.png"))

    # -- Save report --
    with open(os.path.join(output_dir, "analysis_report.txt"), "a") as f:
        f.write("\n" + "=" * 40 + "\n")  # separator between runs
        f.write(f"image {cover}\n")
        f.write(f"message size: {len(message)} characters | image capacity: {cover.size} characters\n")

        f.write("==== Quantitative Metrics ====\n")
        f.write(f"MSE LSB2    : {mse_lsb2:.4f}\n")
        f.write(f"MSE SILSB2  : {mse_silsb2:.4f}\n")
        f.write(f"PSNR LSB2   : {psnr_lsb2:.2f} dB\n")
        f.write(f"PSNR SILSB2 : {psnr_silsb2:.2f} dB\n\n")

        f.write("==== Time Analysis ====\n")
        f.write(f"Embed Time LSB2    : {embed_time_lsb2:.5f} sec\n")
        f.write(f"Embed Time SILSB2  : {embed_time_silsb2:.5f} sec\n")
        f.write(f"Extract Time LSB2  : {extract_time_lsb2:.5f} sec\n")
        f.write(f"Extract Time SILSB2: {extract_time_silsb2:.5f} sec\n\n")

        f.write("==== Message Extraction ====\n")
        f.write(f"LSB2 Extracted Message   : {extracted_lsb2}\n")
        f.write(f"SILSB2 Extracted Message : {extracted_silsb2}\n")


    print("\n[✓] Analysis complete. Results saved in 'output/' folder.")

if __name__ == "__main__":
    main()
