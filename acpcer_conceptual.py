# ACPCER: Adaptive Contextual Predictive Chaining with Entropy Refinement
# Runnable Python Implementation (Conceptual - Token Stream Based)
# Optimized _find_lz_match using bytes.rfind()
# Saves compressed output as .acpcer

import pickle
import os
import sys
import time

# --- Constants and Thresholds ---
MIN_RLE_THRESHOLD = 3
MIN_LZ_THRESHOLD = 4  # Minimum length for an LZ match to be considered "good"
SLIDING_WINDOW_SIZE = 65536 # 64KB. Consider reducing for faster tests if needed (e.g., 8192 for 8KB)
LOOKAHEAD_BUFFER_SIZE = 256 # How many bytes to look ahead for decisions

# --- Control Token Types (used in the token stream) ---
TOKEN_TYPE_RLE = 'RLE'
TOKEN_TYPE_LZ_MATCH = 'LZ'
TOKEN_TYPE_LITERAL = 'LIT'
TOKEN_TYPE_EOF = 'EOF' # End Of File/Stream marker

class EntropyCoder:
    """
    Conceptual "Entropy Coder".
    """
    def __init__(self):
        self.token_stream = []
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
    """
    Conceptual "Entropy Decoder".
    Reads from the token stream produced by EntropyCoder.
    """
    def __init__(self, token_stream):
        self.token_stream = iter(token_stream) # Make it an iterator
        print("Conceptual TokenStream EntropyDecoder initialized.")

    def decode_next_token(self):
        try:
            return next(self.token_stream)
        except StopIteration:
            return (TOKEN_TYPE_EOF,) 

class ACPCER_Compressor:
    def __init__(self):
        self.sliding_window = bytearray(SLIDING_WINDOW_SIZE)
        self.window_pos = 0
        self.window_filled = 0
        self.entropy_coder = EntropyCoder()
        print("ACPCER_Compressor initialized.")

    def _update_sliding_window(self, byte_sequence):
        for byte in byte_sequence:
            self.sliding_window[self.window_pos] = byte
            self.window_pos = (self.window_pos + 1) % SLIDING_WINDOW_SIZE
            if self.window_filled < SLIDING_WINDOW_SIZE:
                self.window_filled += 1
    
    def get_unrolled_window_data(self):
        """Helper to get the current valid window content as a linear bytes object."""
        if self.window_filled == 0:
            return b""
        
        unrolled_data = bytearray(self.window_filled)
        start_read_idx = (self.window_pos - self.window_filled + SLIDING_WINDOW_SIZE) % SLIDING_WINDOW_SIZE
        
        for i in range(self.window_filled):
            unrolled_data[i] = self.sliding_window[(start_read_idx + i) % SLIDING_WINDOW_SIZE]
            
        return bytes(unrolled_data)

    def _find_rle(self, data, current_pos, lookahead_data):
        if not lookahead_data:
            return 0
        
        current_byte = data[current_pos]
        run_length = 0

        for i in range(len(lookahead_data)):
            if lookahead_data[i] == current_byte:
                run_length += 1
            else:
                break
        
        if run_length == len(lookahead_data): 
            for i in range(current_pos + run_length, len(data)):
                if data[i] == current_byte:
                    run_length += 1
                else:
                    break
        
        return run_length if run_length >= MIN_RLE_THRESHOLD else 0

    def _find_lz_match(self, lookahead_data):
        best_len = 0
        best_offset = 0

        if not lookahead_data or self.window_filled == 0:
            return 0, 0

        window_content = self.get_unrolled_window_data()
        if not window_content: 
            return 0,0

        max_search_len = min(len(lookahead_data), len(window_content), LOOKAHEAD_BUFFER_SIZE)

        for length_to_try in range(max_search_len, MIN_LZ_THRESHOLD - 1, -1):
            if length_to_try == 0: continue 
            
            sequence_to_match = lookahead_data[:length_to_try]
            found_idx = window_content.rfind(sequence_to_match)
            
            if found_idx != -1:
                offset = len(window_content) - found_idx
                best_len = length_to_try
                best_offset = offset
                return best_offset, best_len 

        return 0, 0 

    def compress(self, data: bytes):
        print(f"Starting compression of {len(data)} bytes...")
        start_time = time.time()
        current_pos = 0
        progress_update_interval = max(1, len(data) // 100) 

        while current_pos < len(data):
            if current_pos % progress_update_interval == 0 or current_pos == len(data) -1 :
                percentage = (current_pos + 1) / len(data) * 100 if len(data) > 0 else 100
                sys.stdout.write(f"\rCompressing... {current_pos + 1}/{len(data)} bytes ({percentage:.1f}%)")
                sys.stdout.flush()

            lookahead_end = min(current_pos + LOOKAHEAD_BUFFER_SIZE, len(data))
            lookahead_data = data[current_pos : lookahead_end]

            if not lookahead_data:
                break 

            rle_len = self._find_rle(data, current_pos, lookahead_data)
            if rle_len > 0:
                symbol_to_repeat = data[current_pos]
                self.entropy_coder.encode_rle(symbol_to_repeat, rle_len)
                self._update_sliding_window(data[current_pos : current_pos + rle_len])
                current_pos += rle_len
                continue

            lz_offset, lz_len = self._find_lz_match(lookahead_data)
            if lz_len > 0:
                self.entropy_coder.encode_lz_match(lz_offset, lz_len)
                self._update_sliding_window(data[current_pos : current_pos + lz_len])
                current_pos += lz_len
                continue

            current_byte = data[current_pos]
            self.entropy_coder.encode_literal(current_byte)
            self._update_sliding_window(bytes([current_byte]))
            current_pos += 1
        
        sys.stdout.write(f"\rCompressing... {len(data)}/{len(data)} bytes (100.0%) - Finalizing...\n")
        sys.stdout.flush()
        
        self.entropy_coder.encode_eof()
        end_time = time.time()
        print(f"Compression finished in {end_time - start_time:.2f} seconds.")
        return self.entropy_coder.get_token_stream()

class ACPCER_Decompressor:
    def __init__(self):
        self.sliding_window = bytearray(SLIDING_WINDOW_SIZE)
        self.window_pos = 0
        self.window_filled = 0
        print("ACPCER_Decompressor initialized.")

    def _update_sliding_window(self, byte_sequence):
        for byte in byte_sequence:
            self.sliding_window[self.window_pos] = byte
            self.window_pos = (self.window_pos + 1) % SLIDING_WINDOW_SIZE
            if self.window_filled < SLIDING_WINDOW_SIZE:
                self.window_filled += 1

    def decompress(self, token_stream):
        print("Starting decompression...")
        start_time = time.time()
        output_data = bytearray()
        decoder = EntropyDecoder(token_stream)

        while True:
            token = decoder.decode_next_token()
            token_type = token[0]

            if token_type == TOKEN_TYPE_RLE:
                _, symbol, length = token
                decompressed_sequence = bytes([symbol] * length)
                output_data.extend(decompressed_sequence)
                self._update_sliding_window(decompressed_sequence)
            
            elif token_type == TOKEN_TYPE_LZ_MATCH:
                _, offset, length = token
                start_copy_idx = (self.window_pos - offset + SLIDING_WINDOW_SIZE) % SLIDING_WINDOW_SIZE
                decompressed_sequence = bytearray(length) 
                for i in range(length):
                    decompressed_sequence[i] = self.sliding_window[(start_copy_idx + i) % SLIDING_WINDOW_SIZE]
                output_data.extend(decompressed_sequence)
                self._update_sliding_window(decompressed_sequence)

            elif token_type == TOKEN_TYPE_LITERAL:
                _, symbol = token
                decompressed_sequence = bytes([symbol]) 
                output_data.extend(decompressed_sequence)
                self._update_sliding_window(decompressed_sequence) 
            
            elif token_type == TOKEN_TYPE_EOF:
                break 
            
            else:
                print(f"Warning: Unknown or unexpected token type encountered: {token_type}", file=sys.stderr)
                if not token_type : break 
        end_time = time.time()
        print(f"Decompression finished in {end_time - start_time:.2f} seconds.")
        return bytes(output_data)

# --- File Operations and Testing ---
def compress_file(input_filepath, output_acpcer_filepath):
    """Compresses a file and saves the token stream to an .acpcer file."""
    compressor = ACPCER_Compressor()
    try:
        with open(input_filepath, 'rb') as f_in:
            data = f_in.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_filepath}' not found.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading input file '{input_filepath}': {e}", file=sys.stderr)
        return False

    if not data:
        print(f"Input file '{input_filepath}' is empty. Creating empty .acpcer file.", file=sys.stderr)
        with open(output_acpcer_filepath, 'wb') as f_out:
             pickle.dump([(TOKEN_TYPE_EOF,)], f_out) # Save a stream with only EOF
        return True

    token_stream = compressor.compress(data)

    try:
        with open(output_acpcer_filepath, 'wb') as f_out:
            pickle.dump(token_stream, f_out)
        print(f"Compressed token stream saved to: {output_acpcer_filepath}")
        return True
    except Exception as e:
        print(f"Error writing token stream to '{output_acpcer_filepath}': {e}", file=sys.stderr)
        return False

def decompress_file(input_acpcer_filepath, output_filepath):
    """Decompresses an .acpcer token stream file and saves the reconstructed data."""
    decompressor = ACPCER_Decompressor()
    try:
        with open(input_acpcer_filepath, 'rb') as f_in:
            token_stream = pickle.load(f_in)
    except FileNotFoundError:
        print(f"Error: ACPCER file '{input_acpcer_filepath}' not found.", file=sys.stderr)
        return False
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle token stream from '{input_acpcer_filepath}'. File might be corrupted or not a valid .acpcer file.", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Error reading .acpcer file '{input_acpcer_filepath}': {e}", file=sys.stderr)
        return False
    
    if not token_stream : 
        print(f"ACPCER file '{input_acpcer_filepath}' is empty or invalid. Writing empty output.", file=sys.stderr)
        with open(output_filepath, 'wb') as f_out:
            f_out.write(b'')
        return True
    if token_stream == [(TOKEN_TYPE_EOF,)]:
        print(f"ACPCER file indicates an empty original file. Writing empty output.")
        with open(output_filepath, 'wb') as f_out:
            f_out.write(b'')
        return True

    decompressed_data = decompressor.decompress(token_stream)

    try:
        with open(output_filepath, 'wb') as f_out:
            f_out.write(decompressed_data)
        print(f"Decompressed data saved to: {output_filepath}")
        return True
    except Exception as e:
        print(f"Error writing decompressed data to '{output_filepath}': {e}", file=sys.stderr)
        return False

def verify_files(file1_path, file2_path):
    try:
        with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
            data1 = f1.read()
            data2 = f2.read()
        if data1 == data2:
            print(f"Verification SUCCESS: '{file1_path}' and '{file2_path}' are identical.")
            return True
        else:
            print(f"Verification FAILED: '{file1_path}' and '{file2_path}' differ.", file=sys.stderr)
            if abs(len(data1) - len(data2)) > 200 or len(data1) > 5000: 
                 print(f"Length mismatch: original={len(data1)}, decompressed={len(data2)}", file=sys.stderr)
            else: 
                for i in range(min(len(data1), len(data2))):
                    if data1[i] != data2[i]:
                        print(f"Differ at byte {i}: original={data1[i]} (char: {chr(data1[i]) if 32 <= data1[i] <= 126 else 'NP'}), decompressed={data2[i]} (char: {chr(data2[i]) if 32 <= data2[i] <= 126 else 'NP'})", file=sys.stderr)
                        break
                if len(data1) != len(data2):
                    print(f"Length mismatch: original={len(data1)}, decompressed={len(data2)}", file=sys.stderr)
            return False
    except FileNotFoundError:
        print(f"Verification error: One or both files not found ('{file1_path}', '{file2_path}').", file=sys.stderr)
        return False
    except Exception as e:
        print(f"Verification error: {e}", file=sys.stderr)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python <script_name>.py <input_file_path>")
        sample_filename = "sample_input.txt"
        if not os.path.exists(sample_filename):
            try:
                with open(sample_filename, "w", encoding="utf-8") as f: 
                    f.write("This is a test file for the ACPCER compressor. " * 50)
                    f.write("It contains some repetition like AAAAAAAA and BBBBBBBB. ")
                    f.write("Also, some unique sequences like 1234567890. " * 20)
                    f.write("The compressor should try to find RLE, LZ matches, and encode literals. " * 30)
                print(f"Created a sample file: {sample_filename}. Please use it as an argument or provide your own file.")
            except Exception as e:
                print(f"Could not create sample file: {e}")
        sys.exit(1)

    input_file = sys.argv[1]
    
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    base_name = os.path.splitext(input_file)[0]
    original_extension = os.path.splitext(input_file)[1] 
    
    # Define the compressed file extension
    compressed_file_extension = ".acpcer"
    compressed_file = base_name + compressed_file_extension
    
    # Define the decompressed file name, keeping the original extension
    decompressed_file = base_name + "_decompressed" + original_extension

    print(f"\n--- Starting ACPCER Test for: {input_file} ---")

    print("\nStep 1: Compressing...")
    if not compress_file(input_file, compressed_file): # Use new compressed_file name
        print("Compression step failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("\nStep 2: Decompressing...")
    if not decompress_file(compressed_file, decompressed_file): # Use new compressed_file name
        print("Decompression step failed. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("\nStep 3: Verifying...")
    verification_ok = verify_files(input_file, decompressed_file)

    print("\n--- Statistics ---")
    original_size = os.path.getsize(input_file) if os.path.exists(input_file) else 0
    # Get size of the .acpcer file
    acpcer_file_size = os.path.getsize(compressed_file) if os.path.exists(compressed_file) else 0
    
    print(f"Original file size: {original_size} bytes")
    print(f"Compressed '.acpcer' file size: {acpcer_file_size} bytes (Pickled Python objects)")

    if original_size > 0 and acpcer_file_size > 0 :
        conceptual_ratio = original_size / acpcer_file_size
        print(f"Conceptual 'Ratio' (Original Size / Pickled '.acpcer' File Size): {conceptual_ratio:.2f}:1")
    elif original_size == 0:
        print("Original file was empty.")
    else:
        print("Cannot calculate conceptual ratio (one or both file sizes are zero or invalid).")

    if verification_ok:
        print("\nACPCER conceptual test completed successfully!")
    else:
        print("\nACPCER conceptual test FAILED verification.", file=sys.stderr)
