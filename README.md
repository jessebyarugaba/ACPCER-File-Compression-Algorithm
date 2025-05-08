# Understanding the ACPCER Conceptual Compressor

## Introduction

ACPCER (Adaptive Contextual Predictive Chaining with Entropy Refinement) is a conceptual compression algorithm designed to illustrate how different compression techniques can be combined. This document explains a Python implementation of ACPCER.

**Important Note for Beginners:** This script is an *educational tool* to demonstrate compression logic. It does **not** produce truly small compressed files like commercial tools (WinZip, 7-Zip) or even standard libraries (like `gzip`). Instead of outputting a highly optimized stream of bits, it outputs a "token stream" – a list of instructions that represent the compression operations. This token stream is saved as an `.acpcer` file using Python's `pickle` module, which is not a compact format itself.

The main goals of this script are:
1.  To show how a compressor might decide between different strategies (like finding repetitions or identical sequences).
2.  To provide a working compressor and decompressor pair that can correctly reconstruct the original file from this token stream.
3.  To be understandable, even if it means sacrificing some of the extreme complexity and performance of real-world compressors.

## How the Code Works: A High-Level Overview

Imagine you have a long text message you want to make shorter before sending it. You might look for ways to represent parts of it more efficiently:

1.  **Repetitions (Run-Length Encoding - RLE):** If you see "AAAAAAA", you could just write "A repeated 7 times".
2.  **Previously Seen Phrases (Lempel-Ziv - LZ):** If you wrote "the quick brown fox" earlier and you see it again, you could say "repeat the phrase that started X words ago and was Y words long".
3.  **Normal Characters (Literals):** If a character or word is unique or doesn't fit the above, you just write it as is.

The ACPCER script does something similar, but at the level of individual bytes in a file:

* **Input:** It takes any file (e.g., a `.txt` file).
* **Processing:** It reads the file byte by byte. For each byte (or sequence of bytes), it decides:
    * Is this the start of a long run of identical bytes (RLE)?
    * Is this sequence something we've seen recently (LZ match)?
    * If neither, treat this byte as a unique character (Literal).
* **Output (Conceptual Compression):** Instead of writing a super-optimized binary file, it creates a list of "tokens". Each token is like an instruction:
    * `('RLE', byte_value, count)`: Means "this byte `byte_value` is repeated `count` times."
    * `('LZ', distance_back, length)`: Means "copy `length` bytes from `distance_back` bytes ago in the output."
    * `('LIT', byte_value)`: Means "this is the literal byte `byte_value`."
    This list of tokens is saved as an `.acpcer` file.
* **Decompression:** The decompressor reads the `.acpcer` file token by token and follows the instructions to perfectly rebuild the original file.

## Detailed Code Explanation

Let's break down the Python script section by section.

### 1. Imports and Constants

```python
import pickle # For saving/loading the token stream (list of Python objects)
import os     # For file operations like getting file size, checking existence
import sys    # For command-line arguments and exiting the script
import time   # For timing how long compression/decompression takes

# --- Constants and Thresholds ---
MIN_RLE_THRESHOLD = 3       # Shortest run of identical bytes to consider for RLE
MIN_LZ_THRESHOLD = 4        # Shortest sequence to consider for an LZ match
SLIDING_WINDOW_SIZE = 65536 # How many recent bytes to remember for LZ matches (64KB)
LOOKAHEAD_BUFFER_SIZE = 256 # How many bytes ahead to check when making decisions

# --- Control Token Types (used in the token stream) ---
TOKEN_TYPE_RLE = 'RLE'
TOKEN_TYPE_LZ_MATCH = 'LZ'
TOKEN_TYPE_LITERAL = 'LIT'
TOKEN_TYPE_EOF = 'EOF' # End Of File/Stream marker

class EntropyCoder:
    def __init__(self):
        self.token_stream = [] # This list will store our "compressed" output
        print("Conceptual TokenStream EntropyCoder initialized.")

    def encode_rle(self, symbol, length):
        self.token_stream.append((TOKEN_TYPE_RLE, symbol, length))

    def encode_lz_match(self, offset, length):
        self.token_stream.append((TOKEN_TYPE_LZ_MATCH, offset, length))

    def encode_literal(self, symbol):
        self.token_stream.append((TOKEN_TYPE_LITERAL, symbol))

    def encode_eof(self):
        self.token_stream.append((TOKEN_TYPE_EOF,))

    def get_token_stream(self):
        return self.token_stream

class EntropyDecoder:
    def __init__(self, token_stream):
        self.token_stream = iter(token_stream) # Makes it easy to get tokens one by one
        print("Conceptual TokenStream EntropyDecoder initialized.")

    def decode_next_token(self):
        try:
            return next(self.token_stream) # Get the next token from the iterator
        except StopIteration:
            # This case should ideally be handled by an explicit EOF token in the stream,
            # but this is a fallback if the iterator is exhausted.
            return (TOKEN_TYPE_EOF,)
```


### 5. `ACPCER_Compressor` Class

This is the most complex class and contains the core logic for compressing the data.

```python
class ACPCER_Compressor:
    def __init__(self):
        # Sliding window: a circular buffer to store recent output bytes
        self.sliding_window = bytearray(SLIDING_WINDOW_SIZE)
        self.window_pos = 0         # Next position to write in the sliding_window
        self.window_filled = 0      # How many bytes in the window are actual data
        self.entropy_coder = EntropyCoder() # Our simplified token recorder
        print("ACPCER_Compressor initialized.")

    def _update_sliding_window(self, byte_sequence):
        """Adds a sequence of bytes to the sliding window, updating window_pos and window_filled."""
        for byte in byte_sequence:
            self.sliding_window[self.window_pos] = byte
            self.window_pos = (self.window_pos + 1) % SLIDING_WINDOW_SIZE # Modulo for circular behavior
            if self.window_filled < SLIDING_WINDOW_SIZE:
                self.window_filled += 1

    def get_unrolled_window_data(self):
        """Helper to get the current valid window content as a linear bytes object."""
        if self.window_filled == 0:
            return b"" # Return empty bytes if window is empty
        
        # Create a new bytearray for the unrolled data
        unrolled_data = bytearray(self.window_filled)
        # Calculate the actual start index in the circular buffer from where to read
        start_read_idx = (self.window_pos - self.window_filled + SLIDING_WINDOW_SIZE) % SLIDING_WINDOW_SIZE
        
        # Copy data from the circular sliding_window to the linear unrolled_data
        for i in range(self.window_filled):
            unrolled_data[i] = self.sliding_window[(start_read_idx + i) % SLIDING_WINDOW_SIZE]
            
        return bytes(unrolled_data) # Convert to immutable bytes for searching

    def _find_rle(self, data, current_pos, lookahead_data):
        """
        Checks for Run-Length Encoding opportunity from current_pos.
        Returns run length (0 if no RLE opportunity meeting threshold).
        """
        if not lookahead_data: # No data to check
            return 0
        
        current_byte = data[current_pos] # The byte we are checking for repetitions
        run_length = 0

        # Check within the lookahead buffer first
        for i in range(len(lookahead_data)):
            if lookahead_data[i] == current_byte:
                run_length += 1
            else:
                break # The run ended
        
        # If the run potentially extends beyond the lookahead buffer, check the main data stream
        if run_length == len(lookahead_data): # If the run filled the entire lookahead
            # Continue checking from where the lookahead buffer ended in the main 'data'
            for i in range(current_pos + run_length, len(data)):
                if data[i] == current_byte:
                    run_length += 1
                else:
                    break # The run ended
        
        # Only return the run_length if it meets our minimum threshold
        return run_length if run_length >= MIN_RLE_THRESHOLD else 0

    def _find_lz_match(self, lookahead_data):
        """
        Searches for the longest match of lookahead_data within the current sliding_window
        using bytes.rfind() for better performance than manual Python loops.
        Returns: (offset, length) of the best match. Offset is distance from current_pos backwards.
                 (0, 0) if no match >= MIN_LZ_THRESHOLD is found.
        """
        best_len = 0
        best_offset = 0

        if not lookahead_data or self.window_filled == 0:
            return 0, 0 # Cannot find a match if there's nothing to look for or no history

        # Get the current content of the sliding window as a linear bytes object ONCE per call
        window_content = self.get_unrolled_window_data()
        if not window_content: # Should be caught by window_filled == 0, but good to double check
            return 0,0

        # Iterate possible match lengths, from longest possible down to MIN_LZ_THRESHOLD
        # Max length to search for is limited by lookahead_data, window_content, and LOOKAHEAD_BUFFER_SIZE
        max_search_len = min(len(lookahead_data), len(window_content), LOOKAHEAD_BUFFER_SIZE)

        for length_to_try in range(max_search_len, MIN_LZ_THRESHOLD - 1, -1):
            # The loop goes from max_search_len down to MIN_LZ_THRESHOLD.
            # e.g., if MIN_LZ_THRESHOLD is 4, it tries lengths max_search_len, max_search_len-1, ..., 4.
            if length_to_try == 0: continue # Should not happen if MIN_LZ_THRESHOLD >= 1
            
            sequence_to_match = lookahead_data[:length_to_try] # The current sequence we're trying to find
            
            # Search for this sequence in the window_content using rfind.
            # bytes.rfind() finds the *last* occurrence of a substring.
            # In LZ77-style compression, finding the last occurrence often corresponds to
            # the shortest offset (most recent match), which can be beneficial.
            found_idx = window_content.rfind(sequence_to_match)
            
            if found_idx != -1: # If the sequence_to_match was found in window_content
                # Match found.
                # Calculate the offset. Offset is the distance from the "end" of window_content
                # (which conceptually is just before the current_pos in the input stream)
                # back to the start of the 'found_idx' where the match began.
                offset = len(window_content) - found_idx
                
                # Since we iterate length_to_try downwards (from longest to shortest valid length),
                # the first match we find will be the longest possible match for that starting position.
                best_len = length_to_try
                best_offset = offset
                return best_offset, best_len # Return this longest match immediately

        return 0, 0 # No suitable match (>= MIN_LZ_THRESHOLD) was found


    def compress(self, data: bytes):
        """
        Compresses the input data using ACPCER logic.
        Outputs a token stream via the EntropyCoder.
        """
        print(f"Starting compression of {len(data)} bytes...")
        start_time = time.time()
        current_pos = 0 # Our current position in the input 'data'
        # Calculate interval for progress updates (roughly 100 updates or every byte for small files)
        progress_update_interval = max(1, len(data) // 100) 

        while current_pos < len(data): # Loop until we've processed all input data
            # Display progress. '\r' carriage return to overwrite the line, flush to ensure it shows.
            if current_pos % progress_update_interval == 0 or current_pos == len(data) -1 :
                percentage = (current_pos + 1) / len(data) * 100 if len(data) > 0 else 100
                sys.stdout.write(f"\rCompressing... {current_pos + 1}/{len(data)} bytes ({percentage:.1f}%)")
                sys.stdout.flush()

            # Prepare lookahead buffer: a slice of data from current_pos onwards
            lookahead_end = min(current_pos + LOOKAHEAD_BUFFER_SIZE, len(data))
            lookahead_data = data[current_pos : lookahead_end]

            if not lookahead_data: # Should be caught by the 'while' condition, but good for safety
                break 

            # Decision Logic - Priority: RLE > LZ > Literal
            # 1. Try Run-Length Encoding
            rle_len = self._find_rle(data, current_pos, lookahead_data)
            if rle_len > 0: # If a good RLE run was found
                symbol_to_repeat = data[current_pos]
                self.entropy_coder.encode_rle(symbol_to_repeat, rle_len) # Record RLE token
                # Add the actual RLE sequence to our sliding window history
                self._update_sliding_window(data[current_pos : current_pos + rle_len])
                current_pos += rle_len # Move past the RLE sequence in the input
                continue # Go back to the start of the while loop for the new 'current_pos'

            # 2. If no RLE, try Lempel-Ziv Match
            lz_offset, lz_len = self._find_lz_match(lookahead_data)
            if lz_len > 0: # If a good LZ match was found
                self.entropy_coder.encode_lz_match(lz_offset, lz_len) # Record LZ token
                # Add the actual matched sequence to our sliding window history
                # (This sequence is from the input data at current_pos)
                self._update_sliding_window(data[current_pos : current_pos + lz_len])
                current_pos += lz_len # Move past the matched sequence in the input
                continue # Go back to the start of the while loop

            # 3. If no RLE and no LZ, encode as a Literal byte
            current_byte = data[current_pos]
            self.entropy_coder.encode_literal(current_byte) # Record Literal token
            self._update_sliding_window(bytes([current_byte])) # Add the single literal byte to window
            current_pos += 1 # Move to the next byte in the input
        
        # After the loop, ensure the progress message shows 100% and a newline
        sys.stdout.write(f"\rCompressing... {len(data)}/{len(data)} bytes (100.0%) - Finalizing...\n")
        sys.stdout.flush()
        
        self.entropy_coder.encode_eof() # Signal the end of the compressed data
        end_time = time.time()
        print(f"Compression finished in {end_time - start_time:.2f} seconds.")
        return self.entropy_coder.get_token_stream() # Return the list of all recorded tokens
```

### 6. `ACPCER_Decompressor` Class

This class is responsible for taking the `token_stream` (produced by the `ACPCER_Compressor` and saved in the `.acpcer` file) and rebuilding the original data. Its structure and logic must precisely mirror the compressor's actions to ensure correct reconstruction.

```python
class ACPCER_Decompressor:
    def __init__(self):
        self.sliding_window = bytearray(SLIDING_WINDOW_SIZE) # Must match compressor's window size
        self.window_pos = 0
        self.window_filled = 0
        # self.context_models = {} # Context models would be needed for a true PPM decompressor
        print("ACPCER_Decompressor initialized.")

    def _update_sliding_window(self, byte_sequence):
        """Identical to the compressor's _update_sliding_window method."""
        for byte in byte_sequence:
            self.sliding_window[self.window_pos] = byte
            self.window_pos = (self.window_pos + 1) % SLIDING_WINDOW_SIZE
            if self.window_filled < SLIDING_WINDOW_SIZE:
                self.window_filled += 1


    def decompress(self, token_stream):
        """
        Decompresses data from the token_stream.
        Returns the reconstructed byte data.
        """
        print("Starting decompression...")
        start_time = time.time()
        output_data = bytearray() # Reconstructed data will be built here
        decoder = EntropyDecoder(token_stream) # To read tokens from the stream

        while True: # Loop until EOF token is encountered
            token = decoder.decode_next_token() # Get the next instruction
            token_type = token[0] # The first element of the token tuple is its type

            if token_type == TOKEN_TYPE_RLE:
                # Token format: (TOKEN_TYPE_RLE, symbol_to_repeat, length_of_run)
                _, symbol, length = token # Unpack the token's data
                # Recreate the sequence of repeated bytes
                decompressed_sequence = bytes([symbol] * length) 
                output_data.extend(decompressed_sequence) # Add to our output
                self._update_sliding_window(decompressed_sequence) # Update our history
            
            elif token_type == TOKEN_TYPE_LZ_MATCH:
                # Token format: (TOKEN_TYPE_LZ_MATCH, offset_from_end_of_window, length_to_copy)
                _, offset, length = token # Unpack
                
                # Calculate where to start copying from in *our* (the decompressor's) sliding window.
                # The offset tells us how far back from our current position (the end of the
                # currently reconstructed data) the matched sequence started.
                start_copy_idx = (self.window_pos - offset + SLIDING_WINDOW_SIZE) % SLIDING_WINDOW_SIZE
                
                decompressed_sequence = bytearray(length) # Prepare to store the copied bytes
                for i in range(length): # Copy byte by byte from our sliding window
                    decompressed_sequence[i] = self.sliding_window[(start_copy_idx + i) % SLIDING_WINDOW_SIZE]
                
                output_data.extend(decompressed_sequence) # Add to our output
                self._update_sliding_window(decompressed_sequence) # Update our history
            
            elif token_type == TOKEN_TYPE_LITERAL:
                # Token format: (TOKEN_TYPE_LITERAL, literal_byte_value)
                _, symbol = token # Unpack
                decompressed_sequence = bytes([symbol]) # The sequence is just the single literal byte
                output_data.extend(decompressed_sequence) # Add to our output
                self._update_sliding_window(decompressed_sequence) # Update our history
            
            elif token_type == TOKEN_TYPE_EOF:
                break # End Of File token encountered, all done!
            
            else: 
                # This part should ideally not be reached if the token stream is valid
                # and was generated correctly by the compressor.
                print(f"Warning: Unknown or unexpected token type encountered: {token_type}", file=sys.stderr)
                if not token_type : break # Safety break if iterator somehow returns None/empty
                # For a robust decompressor, you might raise an error here:
                # raise ValueError(f"Unknown token type encountered: {token_type}")

        end_time = time.time()
        print(f"Decompression finished in {end_time - start_time:.2f} seconds.")
        return bytes(output_data) # Convert the mutable bytearray to immutable bytes before returning

def compress_file(input_filepath, output_acpcer_filepath):
    """Compresses a file and saves the token stream to an .acpcer file."""
    compressor = ACPCER_Compressor() # Create a new compressor instance
    try:
        with open(input_filepath, 'rb') as f_in: # Open input file in read-binary ('rb') mode
            data = f_in.read() # Read all bytes from the file
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.", file=sys.stderr)
        return False # Indicate failure
    except Exception as e:
        print(f"Error reading input file '{input_filepath}': {e}", file=sys.stderr)
        return False

    if not data: # Handle empty input file
        print(f"Input file '{input_filepath}' is empty. Creating empty .acpcer file.", file=sys.stderr)
        # If input is empty, output a token stream that only contains EOF
        with open(output_acpcer_filepath, 'wb') as f_out: # 'wb' for write-binary
             pickle.dump([(TOKEN_TYPE_EOF,)], f_out) # Save a list with just the EOF token
        return True

    token_stream = compressor.compress(data) # Perform the actual compression

    try:
        # Save the token_stream (which is a Python list of tuples) to the output file
        # using the 'pickle' module. 'wb' means write in binary mode.
        with open(output_acpcer_filepath, 'wb') as f_out: 
            pickle.dump(token_stream, f_out) 
        print(f"Compressed token stream saved to: {output_acpcer_filepath}")
        return True # Indicate success
    except Exception as e:
        print(f"Error writing token stream to '{output_acpcer_filepath}': {e}", file=sys.stderr)
        return False

def decompress_file(input_acpcer_filepath, output_filepath):
    """Decompresses an .acpcer token stream file and saves the reconstructed data."""
    decompressor = ACPCER_Decompressor() # Create a decompressor instance
    try:
        # Open the .acpcer file in read-binary ('rb') mode
        with open(input_acpcer_filepath, 'rb') as f_in: 
            token_stream = pickle.load(f_in) # Load the Python object (our list of tokens)
    except FileNotFoundError:
        print(f"Error: ACPCER file '{input_acpcer_filepath}' not found.", file=sys.stderr)
        return False
    except pickle.UnpicklingError: # Handle if the file is not a valid pickle file
        print(f"Error: Could not unpickle token stream from '{input_acpcer_filepath}'. File might be corrupted or not a valid .acpcer file.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading .acpcer file '{input_acpcer_filepath}': {e}", file=sys.stderr)
        return False
    
    # Handle cases where the loaded token stream might be empty or malformed
    if not token_stream : 
        print(f"ACPCER file '{input_acpcer_filepath}' is empty or invalid. Writing empty output.", file=sys.stderr)
        with open(output_filepath, 'wb') as f_out: f_out.write(b'') 
        return True
    # Specifically check if it's just an EOF token (meaning original was empty)
    if token_stream == [(TOKEN_TYPE_EOF,)]:
        print(f"ACPCER file indicates an empty original file. Writing empty output.")
        with open(output_filepath, 'wb') as f_out: f_out.write(b'')
        return True

    decompressed_data = decompressor.decompress(token_stream) # Perform actual decompression

    try:
        # Write the reconstructed bytes to the output file in write-binary ('wb') mode
        with open(output_filepath, 'wb') as f_out: 
            f_out.write(decompressed_data)
        print(f"Decompressed data saved to: {output_filepath}")
        return True
    except Exception as e:
        print(f"Error writing decompressed data to '{output_filepath}': {e}", file=sys.stderr)
        return False

def verify_files(file1_path, file2_path):
    """Compares two files byte by byte."""
    try:
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            data1 = f1.read() # Read all bytes from first file
            data2 = f2.read() # Read all bytes from second file
        if data1 == data2: # Python can directly compare bytes objects
            print(f"Verification SUCCESS: '{file1_path}' and '{file2_path}' are identical.")
            return True
        else:
            print(f"Verification FAILED: '{file1_path}' and '{file2_path}' differ.", file=sys.stderr)
            # Optional: Print more details about where they differ for small files
            if abs(len(data1) - len(data2)) > 200 or len(data1) > 5000: 
                 print(f"Length mismatch: original={len(data1)}, decompressed={len(data2)}", file=sys.stderr)
            else: 
                for i in range(min(len(data1), len(data2))):
                    if data1[i] != data2[i]:
                        char1 = chr(data1[i]) if 32 <= data1[i] <= 126 else 'NP' # NP for Non-Printable
                        char2 = chr(data2[i]) if 32 <= data2[i] <= 126 else 'NP'
                        print(f"Differ at byte {i}: original={data1[i]} (char: {char1}), decompressed={data2[i]} (char: {char2})", file=sys.stderr)
                        break # Show only the first difference
                if len(data1) != len(data2): # Check length if content was same up to min_len
                    print(f"Length mismatch: original={len(data1)}, decompressed={len(data2)}", file=sys.stderr)
            return False
    except FileNotFoundError:
        print(f"Verification error: One or both files not found ('{file1_path}', '{file2_path}').", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Verification error: {e}", file=sys.stderr)
        return False


if __name__ == "__main__":
    # Check if a command-line argument (input file path) is provided
    if len(sys.argv) < 2: # sys.argv is a list of command-line arguments; sys.argv[0] is script name
        print("Usage: python <script_name>.py <input_file_path>")
        # Create a dummy sample_input.txt for easy testing if no argument is given
        sample_filename = "sample_input.txt"
        if not os.path.exists(sample_filename):
            try:
                with open(sample_filename, "w", encoding="utf-8") as f: # Ensure utf-8 for sample
                    f.write("This is a test file for the ACPCER compressor. " * 50)
                    # ... (rest of sample file content)
                print(f"Created a sample file: {sample_filename}. Please use it as an argument or provide your own file.")
            except Exception as e:
                print(f"Could not create sample file: {e}")
        sys.exit(1) # Exit if no input file is specified by the user

    input_file = sys.argv[1] # Get the input file path from the first command-line argument
    
    # Basic file existence check
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Determine output filenames
    base_name = os.path.splitext(input_file)[0] # Gets the filename without its extension (e.g., "myfile" from "myfile.txt")
    original_extension = os.path.splitext(input_file)[1] # Gets the original extension (e.g., ".txt")
    
    # Define the compressed file extension
    compressed_file_extension = ".acpcer"
    compressed_file = base_name + compressed_file_extension # e.g., "myfile.acpcer"
    
    # Define the decompressed file name, attempting to keep the original extension
    decompressed_file = base_name + "_decompressed" + original_extension # e.g., "myfile_decompressed.txt"

    print(f"\n--- Starting ACPCER Test for: {input_file} ---")

    # Step 1: Compress the input file
    print("\nStep 1: Compressing...")
    if not compress_file(input_file, compressed_file): # Call our compression function
        print("Compression step failed. Exiting.", file=sys.stderr)
        sys.exit(1) # Exit if compression fails

    # Step 2: Decompress the .acpcer file
    print("\nStep 2: Decompressing...")
    if not decompress_file(compressed_file, decompressed_file): # Call our decompression function
        print("Decompression step failed. Exiting.", file=sys.stderr)
        sys.exit(1) # Exit if decompression fails

    # Step 3: Verify that the original and decompressed files are identical
    print("\nStep 3: Verifying...")
    verification_ok = verify_files(input_file, decompressed_file) # Call our verification function

    # Step 4: Print Statistics
    print("\n--- Statistics ---")
    original_size = os.path.getsize(input_file) if os.path.exists(input_file) else 0
    # Get size of the .acpcer file
    acpcer_file_size = os.path.getsize(compressed_file) if os.path.exists(compressed_file) else 0
    
    print(f"Original file size: {original_size} bytes")
    print(f"Compressed '.acpcer' file size: {acpcer_file_size} bytes (Pickled Python objects)")

    # Calculate and print the conceptual ratio
    # This ratio is NOT a true compression ratio because the token stream is not bit-packed.
    # It's a ratio of raw data size to the size of the pickled list of operations.
    if original_size > 0 and acpcer_file_size > 0 :
        conceptual_ratio = original_size / acpcer_file_size
        print(f"Conceptual 'Ratio' (Original Size / Pickled '.acpcer' File Size): {conceptual_ratio:.2f}:1")
    elif original_size == 0: # Handle case of empty original file
        print("Original file was empty.")
    else:
        print("Cannot calculate conceptual ratio (one or both file sizes are zero or invalid).")

    # Final status message based on verification
    if verification_ok:
        print("\nACPCER conceptual test completed successfully!")
    else:
        print("\nACPCER conceptual test FAILED verification.", file=sys.stderr)
```

### How to Run the Script

1.  **Save the Code:** Save the entire Python script (provided in the next section) as a `.py` file (e.g., `acpcer_script.py`).
2.  **Open a Terminal or Command Prompt:** Navigate (using `cd`) to the directory where you saved the Python file.
3.  **Run:** To use a sample file (the script will create `sample_input.txt` in the current directory if it doesn't exist and you run the script without arguments the first time):

    ```bash
    python acpcer_script.py
    ```
    Or, if you have a specific input file:
    ```bash
    python acpcer_script.py your_input_file.txt
    ```