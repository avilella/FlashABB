#!/usr/bin/env python3
import argparse
import os
import sys
import csv
import hashlib

def log(msg, verbose=False, is_verbose_msg=False):
    """
    Prints logging information to sys.stderr.
    If is_verbose_msg is True, it only prints if verbose=True.
    """
    if not is_verbose_msg or verbose:
        print(msg, file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Calculate FlashTAP developability measures for heavy and light chain sequences.")
    parser.add_argument('-i', '--inputfile', required=True, help="Input CSV file containing sequence data.")
    parser.add_argument('--tag', default='fltp', help="Output file tag (default: fltp)")
    parser.add_argument('--outdir', default=None, help="Output directory (default: same as input file)")
    parser.add_argument('--verbose', action='store_true', help="Print detailed processing logs to stderr")
    parser.add_argument('--refresh', action='store_true', help="Force recalculation even if the output file already exists")

    args = parser.parse_args()

    # Determine input paths
    input_file = args.inputfile
    if not os.path.isfile(input_file):
        log(f"Error: Input file '{input_file}' does not exist.", args.verbose)
        sys.exit(1)

    input_dir, input_basename = os.path.split(input_file)
    input_name, _ = os.path.splitext(input_basename)

    # Determine output paths
    out_dir = args.outdir if args.outdir is not None else input_dir
    if out_dir and not os.path.exists(out_dir):
        log(f"Creating output directory: {out_dir}", args.verbose, is_verbose_msg=True)
        os.makedirs(out_dir, exist_ok=True)

    out_filename = f"{input_name}.{args.tag}.csv"
    out_filepath = os.path.join(out_dir, out_filename) if out_dir else out_filename
    out_filepath = os.path.abspath(out_filepath)

    # Check if output already exists and we shouldn't refresh
    if not args.refresh:
        if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 0:
            log("Output file already exists and is non-empty. Skipping calculation.", args.verbose, is_verbose_msg=True)
            # Print strictly the path to STDOUT and exit
            print(out_filepath)
            sys.exit(0)

    # Read input CSV
    log(f"Reading input file: {input_file}", args.verbose, is_verbose_msg=True)
    data = []
    with open(input_file, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = set(reader.fieldnames or [])
        
        # Dynamically determine the column names based on the headers
        name_col = 'name' if 'name' in fieldnames else 'Sequence_Id' if 'Sequence_Id' in fieldnames else None
        heavy_col = 'heavy' if 'heavy' in fieldnames else 'H_Full' if 'H_Full' in fieldnames else None
        light_col = 'light' if 'light' in fieldnames else 'L_Full' if 'L_Full' in fieldnames else None

        if not (name_col and heavy_col and light_col):
            log("Error: Input CSV must contain columns for name ('name' or 'Sequence_Id'), "
                "heavy chain ('heavy' or 'H_Full'), and light chain ('light' or 'L_Full').", args.verbose)
            sys.exit(1)

        for row in reader:
            data.append({
                'name': row[name_col].strip(),
                'heavy': row[heavy_col].strip(),
                'light': row[light_col].strip()
            })

    if not data:
        log("Error: Input CSV is empty.", args.verbose)
        sys.exit(1)

    # Prepare sequences for FlashTAP (Format: HEAVY|LIGHT)
    seqs_for_tap = []
    for item in data:
        seqs_for_tap.append(f"{item['heavy']}|{item['light']}")

    log(f"Loaded {len(seqs_for_tap)} sequences for processing.", args.verbose)

    # Import and run FlashTAP
    log("Loading FlashTAP model...", args.verbose, is_verbose_msg=True)
    try:
        from flash_abb import pretrained_tap
    except ImportError:
        log("Error: Module 'flash_abb' could not be imported.", args.verbose)
        sys.exit(1)

    tap = pretrained_tap(device='cuda')

    log("Calculating FlashTAP measures...", args.verbose)
    result = tap(seqs_for_tap)

    # Prepare output data
    out_data = []
    for i, item in enumerate(data):
        h_chain = item['heavy']
        l_chain = item['light']
        
        # MD5 sum of concatenated heavy and light chains
        concat_seq = h_chain + l_chain
        md5_hash = hashlib.md5(concat_seq.encode('utf-8')).hexdigest()

        # Base output row (standardized output column names)
        out_row = {
            'name': item['name'],
            'heavy': h_chain,
            'light': l_chain,
            'md5sum': md5_hash
        }

        # Add scores (PSH, PPC, PNC, SFvCSP, etc.)
        for k, v in result.scores[i].items():
            out_row[f'score_{k}'] = v

        # Add flag probabilities
        for k, v in result.flag_probs[i].items():
            out_row[f'prob_{k}'] = v

        # Add any flag probability
        out_row['any_flag_prob'] = result.any_flag_prob[i]

        out_data.append(out_row)

    # Write Output
    log(f"Writing results to: {out_filepath}", args.verbose, is_verbose_msg=True)
    if out_data:
        fieldnames = list(out_data[0].keys())
        with open(out_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_data)

    log("Processing complete.", args.verbose, is_verbose_msg=True)
    
    # Print strictly the path to STDOUT
    print(out_filepath)

if __name__ == "__main__":
    main()
