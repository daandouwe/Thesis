#Set job requirements
#PBS -S /bin/bash
#PBS -qgpu
#PBS -lwalltime=5:00

# Loading modules
module load eb
module load Python/3.5.0
module load CUDA/8.0.44
module load cudnn/8.0-v6.0


# Copy input file to scratch
# cp $HOME/big_input_file "$TMPDIR"

# Create output directory on scratch
# mkdir "$TMPDIR"/output_dir

# Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
# python3 $HOME/my_program.py "$TMPDIR"/big_input_file "$TMPDIR"/output_dir
python3 train.py

# Copy output directory from scratch to home
# cp -r "$TMPDIR"/output_dir $HOME
