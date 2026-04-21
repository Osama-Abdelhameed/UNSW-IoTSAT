#!/usr/bin/env python3
"""

    CCSDS_MC_Frame_Count           (8-bit Master Channel Frame Counter,
                                    increments mod 256, exhibits gaps
                                    under DoS/Jamming and duplicates
                                    under Replay)

    CCSDS_Packet_Sequence_Count    (14-bit Space Packet Sequence Count,
                                    same behaviour rules as above,
                                    scaled to the 0..16383 range)

    CCSDS_APID                     (11-bit Application Process Identifier,
                                    fixed per source satellite, with an
                                    anomalous APID injected during
                                    Spoofing subtypes 'beacon' and
                                    'command')

"""

import argparse
import random
import pandas as pd
import numpy as np


# Nominal APIDs per satellite (matches the CCSDS)
# convention of 11-bit APIDs, using values within the user-defined
# range 0x100-0x7FE).
NOMINAL_APID = {
    'Satellite1': 0x101,
    'Satellite2': 0x102,
    # Backwards-compat aliases
    'sat_1': 0x101,
    'sat_2': 0x102,
}
ANOMALOUS_APID = 0x7FF  # reserved/anomalous; used under spoofing injection

# Attack classes that cause counter gaps (frames lost at the receiver)
GAP_CAUSING_ATTACKS = {
    'dos',
    'jamming',
}

# Attack classes that cause counter duplicates (replayed frames)
DUPLICATE_CAUSING_ATTACKS = {
    'replay',
}

# Spoofing subtypes that cause an anomalous APID injection
APID_SPOOFING_SUBTYPES = {
    'beacon',
    'command',
}


def compute_gap_size(attack_type, attack_subtype, severity, rng):
    """Return the counter increment to apply (1 = no gap, >1 = gap size)."""
    if attack_type not in GAP_CAUSING_ATTACKS:
        return 1
    # Severity in [0,1]; larger severity -> larger expected gap.
    # Gap is drawn from a small geometric-like distribution bounded
    # above to avoid unrealistic counter wrap-arounds.
    base = max(0.2, float(severity) if severity is not None else 0.5)
    # Continuous jamming / flooding are more disruptive
    if attack_subtype in ('continuous', 'barrage', 'flooding'):
        base *= 1.5
    gap = 1 + rng.poisson(2.0 * base)
    return int(min(gap, 12))


def should_duplicate(attack_type, attack_subtype, severity, rng):
    """Return True if this frame replays the previous counter value."""
    if attack_type not in DUPLICATE_CAUSING_ATTACKS:
        return False
    # Not every replay-tagged record is a duplicate; duplication
    # probability scales with severity.
    p = 0.3 + 0.4 * (float(severity) if severity is not None else 0.5)
    return rng.random() < p


def spoofed_apid(attack_type, attack_subtype, rng):
    """Return (apid_value, is_anomalous)."""
    if attack_type == 'spoofing' and attack_subtype in APID_SPOOFING_SUBTYPES:
        # Under beacon/command spoofing, 60% of frames carry the
        # anomalous APID; the rest pass through with the nominal
        # APID to reflect an attacker who cycles between legitimate
        # and injected frames.
        if rng.random() < 0.6:
            return ANOMALOUS_APID, True
    return None, False


def augment(input_csv, output_csv, seed=42):
    print(f"Reading {input_csv} ...")
    df = pd.read_csv(input_csv)
    n = len(df)
    print(f"  {n:,} records loaded")

    # Detect the satellite-id column. Adjust this if your CSV uses a different name.
    sat_col = None
    for candidate in ('Satellite_ID', 'SatelliteID', 'satellite_id', 'Satellite'):
        if candidate in df.columns:
            sat_col = candidate
            break
    if sat_col is None:
        # Fallback: synthesise a satellite id by splitting rows in half
        print("  WARN: no satellite-id column found; assigning sat_1/sat_2 by row half")
        sat_ids = ['sat_1'] * (n // 2) + ['sat_2'] * (n - n // 2)
        df['_SatelliteID_synth'] = sat_ids
        sat_col = '_SatelliteID_synth'

    # Detect attack label columns
    attack_type_col = 'Attack_Type' if 'Attack_Type' in df.columns else None
    attack_subtype_col = 'Attack_Subtype' if 'Attack_Subtype' in df.columns else None
    severity_col = 'Attack_Severity' if 'Attack_Severity' in df.columns else None
    if attack_type_col is None:
        raise KeyError("Attack_Type column not found in CSV.")

    rng = np.random.default_rng(seed)

    # Per-satellite running counters
    mc_counter = {sat: 0 for sat in df[sat_col].unique()}
    psc_counter = {sat: 0 for sat in df[sat_col].unique()}
    last_mc = {sat: 0 for sat in df[sat_col].unique()}
    last_psc = {sat: 0 for sat in df[sat_col].unique()}

    mc_out = np.empty(n, dtype=np.int32)
    psc_out = np.empty(n, dtype=np.int32)
    apid_out = np.empty(n, dtype=np.int32)

    for i in range(n):
        sat = df[sat_col].iat[i]
        attack_type = df[attack_type_col].iat[i] if attack_type_col else 'normal'
        attack_subtype = df[attack_subtype_col].iat[i] if attack_subtype_col else ''
        severity = df[severity_col].iat[i] if severity_col else 0.0

        # Normalise the strings (handle NaN / None)
        attack_type = str(attack_type).lower() if pd.notna(attack_type) else 'normal'
        attack_subtype = str(attack_subtype).lower() if pd.notna(attack_subtype) else ''

        if should_duplicate(attack_type, attack_subtype, severity, rng):
            mc_val = last_mc[sat]
            psc_val = last_psc[sat]
        else:
            gap = compute_gap_size(attack_type, attack_subtype, severity, rng)
            mc_counter[sat] = (mc_counter[sat] + gap) % 256
            psc_counter[sat] = (psc_counter[sat] + gap) % 16384
            mc_val = mc_counter[sat]
            psc_val = psc_counter[sat]

        last_mc[sat] = mc_val
        last_psc[sat] = psc_val
        mc_out[i] = mc_val
        psc_out[i] = psc_val

        # APID computation
        spoof_apid, is_anom = spoofed_apid(attack_type, attack_subtype, rng)
        if is_anom:
            apid_out[i] = spoof_apid
        else:
            apid_out[i] = NOMINAL_APID.get(sat, 0x100)

        if (i + 1) % 50000 == 0:
            print(f"  ... processed {i+1:,} / {n:,}")

    df['CCSDS_MC_Frame_Count'] = mc_out
    df['CCSDS_Packet_Sequence_Count'] = psc_out
    df['CCSDS_APID'] = apid_out

    # Drop the synthetic column if we had to create one
    if sat_col == '_SatelliteID_synth':
        df = df.drop(columns=[sat_col])

    print(f"Writing {output_csv} ...")
    df.to_csv(output_csv, index=False)
    print("Done.")
    print(f"  Output columns: {len(df.columns)}")
    print(f"  New columns: CCSDS_MC_Frame_Count, "
          f"CCSDS_Packet_Sequence_Count, CCSDS_APID")


def main():
    parser = argparse.ArgumentParser(
        description="Append CCSDS protocol sub-field columns to UNSW-IoTSAT")
    parser.add_argument("--input", default="UNSW_IoTSAT.csv",
                        help="Path to the input CSV (base release)")
    parser.add_argument("--output", default="UNSW_IoTSAT_with_CCSDS_fields.csv",
                        help="Path to the output CSV (augmented release)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    augment(args.input, args.output, seed=args.seed)


if __name__ == "__main__":
    main()
