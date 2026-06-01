import os, sys, glob


from run_full_rcmip_protocol import load_and_process_protocol

input_dir = "/div/no-backup-nac/users/masan/GRAFITE/temp_indata/"
protocol_file = os.path.join(input_dir, "rcmip_phase3_protocol_v1.1.0.xlsx")
protocol_df = load_and_process_protocol(protocol_file)
outdir = "out_file_dump"
for index, split_exp_df in protocol_df.items():
    print(index)
    for index2, split_exp_df_lev2 in split_exp_df.items():
        print(index2)
        for index3, row in split_exp_df_lev2.iterrows():
            path_expected = os.path.join(outdir, f"{row['Scenario']}_rcmip_draw_samples_500.csv")
            #print(path_expected)
            if not os.path.exists(path_expected):
                print(f"Experiment {row['Scenario']} is missing")

