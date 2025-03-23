#!/bin/bash

# Default values
MODELS_DIR="/home/daniel/Documents/promising_models"
MODEL_PREFIX="obstacle_model_"
MODEL_RANGE="1-10"
PAPARAZZI_HOME="/home/daniel/Documents/GitHub/paparazzi"
AIRCRAFT="bebop_obstacle_avoid"
DRONE_IP="192.168.42.1"
MODEL_TARGET_PATH="sw/airborne/modules/computer_vision/obstacle_detection/src/models/front_model.c"
RUN_TIME=60

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --models-dir)
      MODELS_DIR="$2"
      shift 2
      ;;
    --model-prefix)
      MODEL_PREFIX="$2"
      shift 2
      ;;
    --model-range)
      MODEL_RANGE="$2"
      shift 2
      ;;
    --paparazzi-home)
      PAPARAZZI_HOME="$2"
      shift 2
      ;;
    --aircraft)
      AIRCRAFT="$2"
      shift 2
      ;;
    --drone-ip)
      DRONE_IP="$2"
      shift 2
      ;;
    --run-time)
      RUN_TIME="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Setup environment variables
export PAPARAZZI_HOME=$PAPARAZZI_HOME
export PAPARAZZI_SRC=$PAPARAZZI_HOME

# Process model range
MODEL_INDICES=()
if [[ $MODEL_RANGE == *-* ]]; then
  START=$(echo $MODEL_RANGE | cut -d'-' -f1)
  END=$(echo $MODEL_RANGE | cut -d'-' -f2)
  for ((i=$START; i<=$END; i++)); do
    MODEL_INDICES+=($i)
  done
else
  IFS=',' read -ra MODEL_INDICES <<< "$MODEL_RANGE"
fi

# Create results directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="model_benchmark_results_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Create summary file
SUMMARY_FILE="${RESULTS_DIR}/benchmark_summary.txt"
echo "CNN Model Benchmark Results - ${TIMESTAMP}" > $SUMMARY_FILE
echo "=================================================" >> $SUMMARY_FILE
echo "Models directory: ${MODELS_DIR}" >> $SUMMARY_FILE
echo "Model indices tested: ${MODEL_INDICES[*]}" >> $SUMMARY_FILE
echo "Paparazzi directory: ${PAPARAZZI_HOME}" >> $SUMMARY_FILE
echo "Aircraft: ${AIRCRAFT}" >> $SUMMARY_FILE
echo "" >> $SUMMARY_FILE

# Run benchmark for each model
for MODEL_IDX in "${MODEL_INDICES[@]}"; do
  echo "================================================="
  echo "Testing model ${MODEL_IDX}"
  echo "================================================="
  
  # Copy model file
  MODEL_FILE="${MODEL_PREFIX}${MODEL_IDX}.c"
  MODEL_PATH="${MODELS_DIR}/${MODEL_FILE}"
  TARGET_PATH="${PAPARAZZI_HOME}/${MODEL_TARGET_PATH}"
  
  if [ ! -f "$MODEL_PATH" ]; then
    echo "Model file ${MODEL_PATH} not found. Skipping."
    continue
  fi
  
  echo "Copying ${MODEL_PATH} to ${TARGET_PATH}"
  cp $MODEL_PATH $TARGET_PATH
  
  # Clean aircraft
  echo "Cleaning aircraft..."
  make -C $PAPARAZZI_HOME -f Makefile.ac AIRCRAFT=$AIRCRAFT clean_ac
  if [ $? -ne 0 ]; then
    echo "Failed to clean aircraft for model ${MODEL_IDX}. Skipping."
    continue
  fi
  
  # Compile firmware
  echo "Compiling firmware..."
  make -C $PAPARAZZI_HOME -f Makefile.ac AIRCRAFT=$AIRCRAFT ap.compile
  if [ $? -ne 0 ]; then
    echo "Failed to compile firmware for model ${MODEL_IDX}. Skipping."
    continue
  fi
  
  # Upload to drone
  echo "Uploading to drone..."
  make -C $PAPARAZZI_HOME -f Makefile.ac AIRCRAFT=$AIRCRAFT ap.upload
  if [ $? -ne 0 ]; then
    echo "Failed to upload firmware for model ${MODEL_IDX}. Skipping."
    continue
  fi
  
  # Run test on drone via telnet
  echo "Running test on drone..."
  LOG_FILE="log_model_${MODEL_IDX}.txt"
  
  # Connect via telnet and run commands
  (
    echo "open ${DRONE_IP}"
    sleep 1
    echo "cd data/ftp/internal_000/paparazzi"
    sleep 1
    echo "killall -9 ap.elf > /dev/null 2>&1"
    sleep 1
    echo "./ap.elf > ${LOG_FILE} &"
    sleep 1
    echo "exit"
  ) | telnet > /dev/null
  
  echo "Running model ${MODEL_IDX} for ${RUN_TIME} seconds..."
  sleep $RUN_TIME
  
  # Kill the application
  (
    echo "open ${DRONE_IP}"
    sleep 1
    echo "killall -9 ap.elf"
    sleep 1
    echo "exit"
  ) | telnet > /dev/null
  
  # Record results
  echo "Model ${MODEL_IDX}: Test completed. Log file: ${LOG_FILE}" >> $SUMMARY_FILE
  echo "Model ${MODEL_IDX} test completed. Log file: ${LOG_FILE}"
done

echo "Benchmark complete!"
echo "Summary written to ${SUMMARY_FILE}"
echo "Please download and analyze the log files from the drone using your preferred FTP client."