#!/bin/bash

# Get the output of squeue command for the current user
squeue_output=$(squeue --me)

# Extract the first column (job IDs) starting from the second row
# awk command explanation:
# NR > 1 - skip the first row (header)
# {print $1} - print only the first column
job_ids=$(echo "$squeue_output" | awk 'NR > 1 {print $1}')

# Check if there are any jobs to cancel
if [ -z "$job_ids" ]; then
    echo "No jobs found to cancel."
    exit 0
fi

# Iterate through each job ID and cancel it
for job_id in $job_ids; do
    echo "Cancelling job: $job_id"
    scancel $job_id
done

echo "All jobs have been cancelled."