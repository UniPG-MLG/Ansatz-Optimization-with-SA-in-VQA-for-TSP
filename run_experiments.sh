#!/bin/bash

# Loop over all *_city directories
for city_dir in Instances/*_cities; do
    # Loop over files inside the city directory
    for test_file in "$city_dir"/*; do
        # Get the test file name only (not the full path)
        test_file_name=$(basename "$test_file")
        
        echo "TEST $city_dir | TEST FILE $test_file_name"

        testfile_path="$city_dir/$test_file_name"

        python main.py --log_system_metrics --run_name "EXPERIMENT | $city_dir | POWELL 1000 | 500 SA iter| ALL PARAMS" \
            --max_iterations 500 --cooling_rate 0.999 --eval_file $testfile_path \
            --qc_params all --optimizers powell --vqa_powell_iter 1000
    done
done
