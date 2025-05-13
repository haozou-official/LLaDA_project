# Are Diffusion Language Models Autoregressive After All?

## Installing the environment

The recommended way to run the code is with an Anaconda/Miniconda environment. First, clone the repository:

```
git clone https://github.com/haozou-official/LLaDA_project.git
```

Then, create a new Anaconda environment and install the dependencies:
```
conda env create -f environment.yaml
```

## Preparing datasets

The dataset for the Sudoku is given in [Link](https://drive.google.com/drive/folders/1TluiZjYl-zLdbxjVmhfWl-WyX_OvD7UW?usp=sharing). Please download the test dataset and put it into: `./sudoku_code/sudoku-test-data.npy`

### Sudoku dataset

- `Sudoku-test-data.npy` contains 0.1M sudoku puzzles. Both the dataset only contains puzzles with the unique solutions. 
- Each example of the dataset contains 325 entries: 
    - First value: Number of filled cells in that example puzzle
    - Rest of 324 values (324 values = 4 values corresponding to each of the cell in $9 \times 9$ puzzle) can be divided in two parts. Each example first contains the information about the cells given in the puzzle and followed by the information about the empty cells of the puzzle. 
    - The information (4 values) about each of the cell in the puzzle is given in the form of (row of the cell, column of the cell, correct value of the cell, strategy/list of strategy that needs to be applied to get the correct value). 
    - Strategy id: (0) the cell is given, (2) the cell is filled with Lone single, (3) Hidden single, (4) Naked pair, (5) Naked Triplet, (6) Locked Candidate, (7) XY Wing,
    (8) Unique Rectangle