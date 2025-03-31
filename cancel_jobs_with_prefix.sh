#!/bin/bash

# Check if a prefix is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <job_prefix>"
    echo "Example: $0 test_ (This will cancel all jobs starting with 'test_')"
    echo "To cancel all jobs, use: $0 all"
    exit 1
fi

# Get the prefix from command line argument
PREFIX=$1

# Get the output of squeue command for the current user
squeue_output=$(squeue --me)

# Extract all job information
job_info=$(echo "$squeue_output" | awk 'NR > 1 {print $1 " " $3}')

# Initialize a variable to store filtered job IDs
filtered_job_ids=""

# Check if we want all jobs or just jobs with the specific prefix
if [ "$PREFIX" = "all" ]; then
    # Extract just the job IDs
    filtered_job_ids=$(echo "$job_info" | awk '{print $1}')
else
    # Filter jobs that match the prefix
    filtered_job_ids=$(echo "$job_info" | awk -v prefix="$PREFIX" '$1 ~ "^" prefix {print $1}')
fi

# Check if there are any jobs to cancel
if [ -z "$filtered_job_ids" ]; then
    echo "No jobs found with prefix '$PREFIX' to cancel."
    exit 0
fi

# Count how many jobs will be cancelled
job_count=$(echo "$filtered_job_ids" | wc -l)
echo "Found $job_count job(s) with prefix '$PREFIX' to cancel."
echo "Jobs to be cancelled:"
echo "$job_info" | awk -v prefix="$PREFIX" '$1 ~ "^" prefix || prefix == "all" {print "JobID: " $1 "}'

# Ask for confirmation
read -p "Do you want to continue with cancellation? (y/n): " confirm
if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
    echo "Cancellation aborted."
    exit 0
fi

# Iterate through each job ID and cancel it
for job_id in $filtered_job_ids; do
    echo "Cancelling job: $job_id"
    scancel $job_id
done

echo "All matching jobs have been cancelled."